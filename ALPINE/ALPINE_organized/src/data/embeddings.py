"""
Protein language model embedding extractors for various PLMs.
Supports ESM-2, Ablang2, ESM-C, PROGEN models.

Each extractor matches the exact model loading and preprocessing used in the
corresponding PLM notebook to ensure reproducible embeddings.
"""

import torch
from typing import List
from tqdm import tqdm


class BaseEmbeddingExtractor:
    """
    Base class for protein language model embedding extraction.
    Uses HuggingFace AutoModel + mean pooling with CLS/CLS separator.
    """

    def __init__(self, model_name: str, max_seq_len: int = 1024, batch_size: int = 16,
                 device: str = "auto", trust_remote_code: bool = False):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)

        print(f"🔧 Initializing {self.__class__.__name__} on {self.device}")
        print(f"📥 Loading model: {model_name}")

        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.cls_token = self.tokenizer.cls_token
        self.embedding_size = self.model.config.hidden_size

        print(f"✅ Model loaded. Embedding size: {self.embedding_size} | CLS Token: {self.cls_token}")

    @torch.no_grad()
    def get_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate mean-pooled embeddings for a list of sequences.
        Processes in batches of self.batch_size.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_embeddings = []
        for start in tqdm(range(0, len(sequences), self.batch_size),
                          desc="Generating Embeddings", leave=False):
            batch_seqs = sequences[start:start + self.batch_size]
            batch_embs = self._embed_batch(batch_seqs)
            for i in range(len(batch_seqs)):
                all_embeddings.append(batch_embs[i:i+1])

        return torch.cat(all_embeddings, dim=0)

    def _embed_batch(self, sequences: List[str]) -> torch.Tensor:
        """Embed a single batch. Override in subclasses for custom preprocessing."""
        processed = self._preprocess_sequences(sequences)
        inputs = self.tokenizer(processed, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state.to(torch.float32)

        # Some tokenizers / model wrappers (custom remote code) may not
        # return an `attention_mask`. Create a sensible fallback mask from
        # `input_ids` if needed, or use an all-ones mask when no pad token
        # is defined.
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is None:
            input_ids = inputs.get('input_ids')
            pad_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_id is not None and input_ids is not None:
                attention_mask = (input_ids != pad_id).long()
            else:
                # No pad token defined — treat every position as valid
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)

        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu()

    def _preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """Replace '|' with CLS+CLS separator (matches ESM-2 notebook)."""
        return [s.replace('|', f"{self.cls_token}{self.cls_token}") for s in sequences]


class ESM2EmbeddingExtractor(BaseEmbeddingExtractor):
    """
    ESM-2 embedding extractor.
    Loads with float16 on GPU, uses CLS+CLS separator, mean pooling.
    Matches ESM_2_CLS notebook exactly.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D",
                 max_seq_len: int = 1024, batch_size: int = 16, device: str = "auto"):
        super().__init__(model_name, max_seq_len, batch_size, device,
                         trust_remote_code=False)


class Ablang2EmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Ablang2 embedding extractor.
    Requires trust_remote_code=True and model "hemantn/ablang2".
    Uses CLS+CLS separator, mean pooling.  Matches ABLANG2_NEW_CLS notebook.

    Note: hemantn/ablang2 uses custom modeling code whose internal imports
    require the cached model directory to be on sys.path.  This constructor
    adds it automatically via huggingface_hub.snapshot_download().
    """

    def __init__(self, model_name: str = "hemantn/ablang2",
                 max_seq_len: int = 1024, batch_size: int = 32, device: str = "auto"):
        import sys
        import os
        import shutil
        # hemantn/ablang2 remote code imports 'configuration_ablang2paired' as a
        # top-level module.  We must put the cached snapshot dir on sys.path.
        # On Windows the HuggingFace symlink workaround can leave some .py files
        # missing from the transformers modules cache — we copy them over too.
        try:
            from huggingface_hub import snapshot_download
            snap_dir = snapshot_download(repo_id=model_name)
            if snap_dir not in sys.path:
                sys.path.insert(0, snap_dir)

            # Sync any .py files missing from the transformers modules cache
            # (Windows symlink limitation workaround).
            snap_id = os.path.basename(snap_dir)
            org, repo = model_name.split("/", 1)
            modules_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "modules",
                "transformers_modules", org, repo, snap_id
            )
            if os.path.isdir(modules_dir):
                for fname in os.listdir(snap_dir):
                    if fname.endswith(".py"):
                        dst = os.path.join(modules_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy2(os.path.join(snap_dir, fname), dst)
        except Exception as e:
            print(f"   Warning: could not prepare {model_name} cache: {e}")

        # Reduce batch size automatically on CUDA devices to avoid OOMs
        # for large custom models like hemantn/ablang2. This keeps the
        # default behavior on CPU unchanged.
        effective_bs = min(batch_size, 8) if torch.cuda.is_available() else batch_size

        super().__init__(model_name, max_seq_len, effective_bs, device,
                 trust_remote_code=True)


class ProgenEmbeddingExtractor:
    """
    PROGEN-2 embedding extractor (small and medium variants).

    ProGen-2 is a causal language model.  Loading uses AutoModelForCausalLM
    with trust_remote_code=True and left-side padding.  The separator between
    chains is the EOS token (not CLS).  Embedding size is n_embd from config.
    Matches PROGEN_SMALL_NEW_CLS and PROGEN_MEDIUM_CLS notebooks exactly.
    """


    def __init__(self, model_name: str = "hugohrban/progen2-small",
                 max_seq_len: int = 1024, batch_size: int = 32, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)

        print(f"🔧 Initializing ProgenEmbeddingExtractor on {self.device}")
        print(f"📥 Loading model: {model_name}")

        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'   # Required for causal LM

        for param in self.model.parameters():
            param.requires_grad = False

        self.sep_token = self.tokenizer.eos_token
        self.embedding_size = self.model.config.n_embd   # ProGen uses n_embd

        print(f"✅ Model loaded. Embedding size: {self.embedding_size} | EOS separator: {self.sep_token!r}")

    @torch.no_grad()
    def get_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate mean-pooled embeddings using last hidden states.
        Replaces '|' with EOS+EOS as multi-chain separator.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_embeddings = []
        for start in tqdm(range(0, len(sequences), self.batch_size),
                          desc="Generating Embeddings", leave=False):
            batch_seqs = sequences[start:start + self.batch_size]
            # Replace | with two EOS tokens (matches notebook exactly)
            processed = [s.replace('|', f"{self.sep_token}{self.sep_token}") for s in batch_seqs]

            inputs = self.tokenizer(processed, padding=True, truncation=True,
                                    max_length=self.max_seq_len, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1].to(torch.float32)  # [B, L, D]

            # Mean pooling with attention mask
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            batch_embs = (sum_emb / sum_mask).cpu()

            for i in range(len(batch_seqs)):
                all_embeddings.append(batch_embs[i:i+1])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0)


class ESMCEmbeddingExtractor:
    """
    ESM-C embedding extractor using the EvolutionaryScale ESM SDK.

    Requires the 'esm' package from EvolutionaryScale (pip install esm).
    Uses ESMC.from_pretrained("esmc_300m") and returns mean-pooled sequence
    embeddings via LogitsConfig(return_embeddings=True).
    Matches ESM_C_CLS notebook exactly.
    
    NOTE: ESM-C requires ESM v2.2.0 or compatible version.
    The current environment may have an incompatible ESM version (v3.2.0+).
    """

    def __init__(self, model_name: str = "esmc_300m",
                 batch_size: int = 32, device: str = "auto"):
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            self._ESMProtein = ESMProtein
            self._LogitsConfig = LogitsConfig
        except ImportError:
            raise ImportError(
                "ESM-C requires the EvolutionaryScale 'esm' package. "
                "Install it with: pip install esm"
            )

        self.model_name = model_name
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else torch.device(device)

        print(f"🔧 Initializing ESMCEmbeddingExtractor on {self.device}")
        print(f"📥 Loading ESM-C model: {model_name}")

        # Try to load the model
        try:
            from esm.models.esmc import ESMC as ESMC_cls
            self.model = ESMC_cls.from_pretrained(model_name).to(self.device)
        except ImportError as e:
            if "load_local_model" in str(e):
                error_msg = (
                    f"❌ ESM-C loading failed: {e}\n\n"
                    f"Root cause: The current ESM package (v3.2.0+) is incompatible with ESMC.\n"
                    f"The from_pretrained() method requires 'load_local_model' which was removed.\n\n"
                    f"To fix this, choose one of the following:\n"
                    f"1. Downgrade ESM to v2.2.0 (compatible version):\n"
                    f"   pip install esm==2.2.0\n"
                    f"2. Or skip ESM-C and use other PLMs (esm2, ablang2, progen2_small, progen2_medium):\n"
                    f"   python train_plms.py --config configs/plms_config.yaml --plm esm2\n\n"
                    f"Skipping ESM-C for this run..."
                )
                raise ImportError(error_msg) from e
            raise

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # Determine embedding size via dummy pass
        dummy = self._ESMProtein(sequence="A")
        dummy_tensor = self.model.encode(dummy)
        dummy_output = self.model.logits(
            dummy_tensor, self._LogitsConfig(sequence=True, return_embeddings=True)
        )
        self.embedding_size = dummy_output.embeddings.shape[-1]

        print(f"✅ ESM-C loaded. Embedding size: {self.embedding_size}")

    @torch.no_grad()
    def get_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate mean-pooled embeddings from ESM-C.
        Processes one sequence at a time (ESM-C API is sequential).
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_embeddings = []
        for seq in tqdm(sequences, desc="Generating Embeddings", leave=False):
            protein = self._ESMProtein(sequence=seq)
            encoded = self.model.encode(protein)
            output = self.model.logits(
                encoded, self._LogitsConfig(sequence=True, return_embeddings=True)
            )
            # output.embeddings shape: [1, seq_len, embedding_size]
            emb = output.embeddings.mean(dim=1).cpu().to(torch.float32)  # [1, D]
            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)


def get_embedding_extractor(plm_key: str, model_name: str = None,
                             batch_size: int = 32, device: str = "auto"):
    """
    Factory function to get the appropriate embedding extractor for a PLM.

    Args:
        plm_key: PLM identifier: 'esm2', 'ablang2', 'esm_c', 'progen2_small',
                 'progen2_medium', or 'custom'.
        model_name: Override the default model name.
        batch_size: Batch size for embedding generation.
        device: Device string.

    Returns:
        Appropriate embedding extractor instance.
    """
    defaults = {
        'esm2':          ('facebook/esm2_t33_650M_UR50D', ESM2EmbeddingExtractor,     16),
        'ablang2':       ('hemantn/ablang2',               Ablang2EmbeddingExtractor,  32),
        'esm_c':         ('esmc_300m',                     ESMCEmbeddingExtractor,     32),
        'progen2_small': ('hugohrban/progen2-small',       ProgenEmbeddingExtractor,   32),
        'progen2_medium':('hugohrban/progen2-medium',      ProgenEmbeddingExtractor,   32),
    }

    key = plm_key.lower()
    if key not in defaults:
        raise ValueError(
            f"Unknown PLM key '{plm_key}'. "
            f"Choose from: {list(defaults.keys())} or use model_name directly."
        )

    default_model, extractor_cls, default_bs = defaults[key]
    name = model_name or default_model
    bs = batch_size or default_bs

    if extractor_cls is ESMCEmbeddingExtractor:
        return extractor_cls(model_name=name, batch_size=bs, device=device)
    return extractor_cls(model_name=name, batch_size=bs, device=device)
