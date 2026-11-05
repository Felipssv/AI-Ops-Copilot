import hashlib
from pathlib import Path
import pandas as pd

from ai_ops_copilot.domain.exceptions import InvalidFileFormatError
from ai_ops_copilot.domain.orders import REQUIRED_ORDER_COLUMNS

def compute_file_sha256(file_path: Path | str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise InvalidFileFormatError(message="O path do arquivo não existe!", 
                                      details={"file_name": path.name})
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_orders_csv(file_path: Path | str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise InvalidFileFormatError(message="O path do arquivo não existe!", 
                                      details={"file_name": path.name})
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        raise InvalidFileFormatError(
            message=f"Não foi possível ler o arquivo CSV '{path.name}'.",
            details={"file_name": path.name, "reason": str(e)},
        ) from e

    if df.empty:
        raise InvalidFileFormatError(message="O arquivo csv não há registros(arquivo vazio)!")

    missing_cols = [col for col in REQUIRED_ORDER_COLUMNS if col not in df.columns]

    if missing_cols:
        raise InvalidFileFormatError(message=f"As seguintes colunas estão faltando: {missing_cols}",
                                     details={"file_name": path.name, " missing_columns": missing_cols})
    return df