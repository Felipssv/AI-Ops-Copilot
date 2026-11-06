from dataclasses import dataclass
import pandas as pd
from ai_ops_copilot.domain.datasets import (
    DatasetQualityReport,
    QualityIssueCode,
    QualityWarning,
)

from ai_ops_copilot.domain.exceptions import InvalidFileFormatError
from ai_ops_copilot.domain.orders import (
    DEFAULT_LATE_TOLERANCE_MINUTES,
    MAX_ACTUAL_MINUTES,
    MAX_ESTIMATED_MINUTES,
    MAX_ORDER_VALUE,
    MIN_ACTUAL_MINUTES,
    MIN_ESTIMATED_MINUTES,
    MIN_ORDER_VALUE,
    REQUIRED_ORDER_COLUMNS,
    VALID_ORDER_CATEGORIES,
    VALID_REGIONS,
)
MIN_VALID_RECORDS_THRESHOLD = 100
MAX_INVALID_RATE_THRESHOLD = 0.20
@dataclass(frozen=True)
class ValidationOutcome:
    report: DatasetQualityReport
    valid_df: pd.DataFrame

def validate_orders_dataframe(
    df: pd.DataFrame,
    dataset_id: str,
    late_tolerance_minutes: int = DEFAULT_LATE_TOLERANCE_MINUTES,
) -> ValidationOutcome:
    missing_cols = [col for col in REQUIRED_ORDER_COLUMNS if col not in df.columns]
    if missing_cols:
        raise InvalidFileFormatError(
            message=f"Colunas obrigatórias ausentes: {missing_cols}",
            details={"missing_columns": missing_cols},
        )

    # Lista de regras executadas: cada item é (campo, codigo_erro, mascara_de_linhas_com_erro)
    checks: list[tuple[str, QualityIssueCode, pd.Series]] = []

    # 1. Valores ausentes (em todas as colunas obrigatórias)
    for col in REQUIRED_ORDER_COLUMNS:
        null_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        checks.append((col, QualityIssueCode.MISSING_VALUES, null_mask))

    # 2. IDs duplicados
    dup_mask = df["order_id"].duplicated(keep="first") & ~df["order_id"].isna()
    checks.append(("order_id", QualityIssueCode.DUPLICATE_ORDER_ID, dup_mask))

    # 3. Data/hora inválida
    parsed_dt = pd.to_datetime(df["created_at"], errors="coerce")
    checks.append(("created_at", QualityIssueCode.INVALID_DATETIME, parsed_dt.isna() & ~df["created_at"].isna()))

    # 4. Categorias válidas (região e categoria agrupadas em dicionário!)
    allowed_categories = {
        "region": VALID_REGIONS,
        "order_category": VALID_ORDER_CATEGORIES,
    }
    for col, valid_set in allowed_categories.items():
        bad_cat_mask = ~df[col].isin(valid_set) & ~df[col].isna()
        checks.append((col, QualityIssueCode.UNKNOWN_CATEGORY, bad_cat_mask))

    # 5. Limites numéricos (todas as 5 colunas numéricas em um único loop!)
    numeric_bounds = {
        "order_value": (MIN_ORDER_VALUE, MAX_ORDER_VALUE),
        "estimated_prep_minutes": (MIN_ESTIMATED_MINUTES, MAX_ESTIMATED_MINUTES),
        "estimated_delivery_minutes": (MIN_ESTIMATED_MINUTES, MAX_ESTIMATED_MINUTES),
        "actual_prep_minutes": (MIN_ACTUAL_MINUTES, MAX_ACTUAL_MINUTES),
        "actual_delivery_minutes": (MIN_ACTUAL_MINUTES, MAX_ACTUAL_MINUTES),
    }
    for col, (min_val, max_val) in numeric_bounds.items():
        num_col = pd.to_numeric(df[col], errors="coerce")
        out_mask = (num_col.isna() | (num_col < min_val) | (num_col > max_val)) & ~df[col].isna()
        checks.append((col, QualityIssueCode.OUT_OF_RANGE, out_mask))

    # Consolidação única de todos os checks até aqui (sem repetir código!)
    warnings: list[QualityWarning] = []
    invalid_mask = pd.Series(False, index=df.index)
    for field, issue, mask in checks:
        count = int(mask.sum())
        if count > 0:
            warnings.append(QualityWarning(field=field, issue=issue, count=count))
            invalid_mask = invalid_mask | mask

    # 6. Consistência da flag is_late (avaliada apenas nas linhas que têm números válidos!)
    est_prep = pd.to_numeric(df["estimated_prep_minutes"], errors="coerce")
    est_deliv = pd.to_numeric(df["estimated_delivery_minutes"], errors="coerce")
    act_prep = pd.to_numeric(df["actual_prep_minutes"], errors="coerce")
    act_deliv = pd.to_numeric(df["actual_delivery_minutes"], errors="coerce")

    expected_late = (act_prep + act_deliv) > (est_prep + est_deliv + late_tolerance_minutes)
    late_mismatch = (df["is_late"].astype(bool) != expected_late) & ~invalid_mask
    if int(late_mismatch.sum()) > 0:
        warnings.append(
            QualityWarning(
                field="is_late",
                issue=QualityIssueCode.INCONSISTENT_LATE_FLAG,
                count=int(late_mismatch.sum()),
            )
        )
        invalid_mask = invalid_mask | late_mismatch

    valid_df = df.loc[~invalid_mask].copy()
    
    total_records = len(df)
    valid_records = len(valid_df)
    invalid_records = int(invalid_mask.sum())
    is_approved = (
        valid_records >= MIN_VALID_RECORDS_THRESHOLD
        and (invalid_records / max(total_records, 1)) <= MAX_INVALID_RATE_THRESHOLD
    )
    report = DatasetQualityReport(
        dataset_id=dataset_id,
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        warnings=warnings,
        status="completed" if is_approved else "failed",
    )
    return ValidationOutcome(report=report, valid_df=valid_df)
