from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal



class RealEstateContract(BaseModel):
    """
    Schema optimized for Guided Decoding in vLLM.
    All fields have a matching column in master_clauses.csv (CUAD benchmark).
    """

    # Force all fields into JSON Schema "required" so xgrammar (vLLM Guided
    # Decoding backend) cannot close the JSON object before generating every
    # field. Without this, Optional fields with defaults are omitted from
    # "required" and xgrammar closes the object early, producing truncated JSON.
    model_config = ConfigDict(
        json_schema_extra=lambda schema: schema.update(
            required=list(schema.get("properties", {}).keys())
        )
    )

    # --- Entities and Dates ---
    parties: str = Field(
        default="Not Mentioned",
        description="Full legal names of the entities or individuals entering the agreement. Use 'Not Mentioned' only if no party names appear in the text."
    )
    agreement_date: Optional[str] = Field(
        default=None,
        description="The date the contract was signed. Keywords: 'dated', 'entered into as of', 'executed on'. Format: YYYY-MM-DD."
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="The date obligations begin — often different from agreement_date. Keywords: 'effective as of', 'commencing on', 'shall commence'. Format: YYYY-MM-DD."
    )
    expiration_date: Optional[str] = Field(
        default=None,
        description="The date the contract ends naturally. Keywords: 'expire', 'expiration date', 'terminate on', 'through and including'. Format: YYYY-MM-DD."
    )

    # --- Operative Clauses ---
    governing_law: Optional[str] = Field(
        default=None,
        description="The jurisdiction whose laws govern the contract (e.g., 'Delaware', 'New York')."
    )
    anti_assignment: Optional[str] = Field(
        default="Not Mentioned",
        description=(
            "Whether the contract restricts assignment (transfer) of the agreement to a third party. "
            "Use 'Prohibited' when assignment is explicitly forbidden without any option for consent "
            "(keywords: 'may not assign', 'shall not assign', 'cannot be assigned', 'non-assignable', "
            "'assignment is prohibited'). "
            "Use 'Allowed with consent' when assignment is permitted only with the other party's prior approval "
            "(keywords: 'prior written consent', 'with consent', 'consent shall not be unreasonably withheld', "
            "'may assign with the consent'). "
            "Use 'Not Mentioned' only when the contract contains no assignment clause at all."
        )
    )
    renewal_term: Optional[str] = Field(
        default=None,
        description=(
            "Duration of each AUTOMATIC renewal period, not the initial term. "
            "Keywords: 'automatically renew', 'successive periods of', 'renewal term', 'renewed for'. "
            "Extract only the duration, e.g. '1 year', '6 months', '30 days'."
        )
    )
    notice_period_to_terminate_renewal: Optional[str] = Field(
        default=None,
        description=(
            "Advance notice required specifically to PREVENT automatic renewal — not a general termination clause. "
            "Keywords: 'prior written notice', 'days before expiration', 'days prior to the end', 'notice of non-renewal'. "
            "Extract only the period, e.g. '30 days', '60 days', '90 days'. "
            "Usually found in the same paragraph as renewal_term."
        )
    )

    # --- Risk and Control Clauses ---
    audit_rights: Optional[Literal["Yes", "No"]] = Field(
        default="No",
        description=(
            "Does any party have the right to audit or inspect the other party's books, records, or financials? "
            "Answer 'Yes' for keywords: 'right to audit', 'audit the books', 'inspect the records', "
            "'books and records', 'right of inspection', 'access to books', 'audit rights'."
        )
    )
    cap_on_liability: Optional[Literal["Yes", "No"]] = Field(
        default="No",
        description=(
            "Is there a maximum dollar limit on one or both parties' total liability or damages? "
            "Answer 'Yes' for keywords: 'shall not exceed', 'aggregate liability', 'in no event shall', "
            "'maximum liability', 'limited to', 'liability cap', 'total liability'."
        )
    )
    termination_for_convenience: Optional[Literal["Yes", "No"]] = Field(
        default="No",
        description=(
            "Can either party terminate the contract without needing to prove a breach or cause? "
            "Answer 'Yes' for keywords: 'terminate for convenience', 'terminate at any time', "
            "'without cause', 'without reason', 'at its sole discretion', 'upon [X] days notice', "
            "'either party may terminate'."
        )
    )
    liquidated_damages: Optional[Literal["Yes", "No"]] = Field(
        default="No",
        description=(
            "Does the contract specify a pre-determined monetary penalty for breach or failure to perform? "
            "Answer 'Yes' for keywords: 'liquidated damages', 'agreed damages', 'stipulated damages', "
            "'penalty of $', 'shall pay [amount] as damages', 'per day for each day uncured'."
        )
    )