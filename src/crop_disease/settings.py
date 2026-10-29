"""Environment-backed settings; secrets must never be committed."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    model_checkpoint: str = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pt")


settings = Settings()
