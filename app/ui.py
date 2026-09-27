import os
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO

import pandas as pd
import streamlit as st
from langchain_core.messages import ToolMessage
from PIL import Image, ImageOps
from pydantic import ValidationError

from app.agent import ask_receipts
from app.receipts import ReceiptDraft, extract_receipt
from app.store import Store


def amount_to_cents(amount: float | None) -> int:
    try:
        cents = Decimal(str(amount)) * 100
    except InvalidOperation as exc:
        raise ValueError("Enter an amount for each total and item.") from exc
    if not cents.is_finite() or cents < 0 or cents > 1_000_000_000:
        raise ValueError("Amounts must be between 0.00 and 10,000,000.00.")
    if cents != cents.to_integral_value():
        raise ValueError("Amounts can have at most two decimal places.")
    return int(cents)


def receipt_details(receipt: dict) -> None:
    title = (f"{receipt['purchased_on']} · {receipt['merchant']} · "
             f"{receipt['currency']} {receipt['total_cents'] / 100:.2f}")
    with st.expander(title):
        st.caption(f"Receipt {receipt['id']}")
        st.table([
            {"Item": item["name"], "Quantity": item["quantity"],
             "Line total": f"{receipt['currency']} {item['line_total_cents'] / 100:.2f}"}
            for item in receipt["items"]
        ])
        st.caption(f"Tax: {receipt['currency']} {receipt['tax_cents'] / 100:.2f}")
        for warning in receipt.get("warnings", []):
            st.warning(warning)
        if receipt.get("raw_text"):
            st.text(receipt["raw_text"])


def upload_and_review(store: Store) -> None:
    st.subheader("Upload a receipt")
    st.caption("Extraction sends the receipt image to the configured model provider. Review its results before saving.")
    uploaded = st.file_uploader("Receipt image", type=["png", "jpg", "jpeg", "webp", "mpo"])
    if st.button("Extract receipt", disabled=uploaded is None) and uploaded is not None:
        try:
            with st.spinner("Reading receipt…"):
                image = uploaded.getvalue()
                st.session_state.draft = extract_receipt(image)
                st.session_state.receipt_image = image
                st.session_state.draft_version = st.session_state.get("draft_version", 0) + 1
        except Exception as exc:
            st.error(str(exc))
    draft: ReceiptDraft | None = st.session_state.get("draft")
    if draft is None:
        st.info("Upload an image to begin. Review the extracted fields before saving.")
        return
    st.subheader("Review extracted fields")
    for warning in draft.warnings:
        st.warning(warning)
    with st.expander("Original receipt"):
        if image := st.session_state.get("receipt_image"):
            with Image.open(BytesIO(image)) as preview:
                st.image(ImageOps.exif_transpose(preview).convert("RGB"))
        st.caption("Model transcription; compare it with the original image.")
        st.text(draft.raw_text)
    with st.form(f"review_{st.session_state.draft_version}"):
        merchant = st.text_input("Merchant", value=draft.merchant or "", max_chars=200)
        purchased_on = st.date_input("Purchase date", value=draft.purchased_on)
        currencies = ["USD", "EUR", "GBP"]
        currency = st.selectbox(
            "Currency", currencies,
            index=currencies.index(draft.currency) if draft.currency in currencies else None,
            placeholder="Choose the receipt currency",
        )
        st.caption("Enter amounts in the selected currency, for example 8.50. Line totals include all units.")
        total = st.number_input(
            "Receipt total", min_value=0.0, max_value=10_000_000.0,
            value=draft.total_cents / 100 if draft.total_cents is not None else None,
            step=0.01, format="%.2f",
        )
        tax = st.number_input(
            "Tax", min_value=0.0, max_value=10_000_000.0,
            value=draft.tax_cents / 100 if draft.tax_cents is not None else None,
            step=0.01, format="%.2f",
        )
        item_rows = pd.DataFrame(
            [item.model_dump() for item in draft.items], columns=["name", "quantity", "line_total_cents"]
        )
        item_rows["line_total"] = item_rows.pop("line_total_cents").astype(float) / 100
        items = st.data_editor(
            item_rows,
            num_rows="dynamic", hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Item", required=True, max_chars=500),
                "quantity": st.column_config.NumberColumn("Quantity", min_value=1, max_value=1_000_000,
                                                          step=1, required=True),
                "line_total": st.column_config.NumberColumn(
                    "Line total", min_value=0.0, max_value=10_000_000.0,
                    step=0.01, format="%.2f", required=True),
            },
        )
        reviewed = st.checkbox("I reviewed the merchant, date, currency, items and amounts")
        submitted = st.form_submit_button("Save receipt")
    if submitted:
        try:
            result = store.save({
                "merchant": merchant, "purchased_on": purchased_on, "currency": currency,
                "total_cents": amount_to_cents(total), "tax_cents": amount_to_cents(tax),
                "items": [
                    {"name": item["name"], "quantity": item["quantity"],
                     "line_total_cents": amount_to_cents(item["line_total"])}
                    for item in items.to_dict("records")
                ],
                "reviewed": reviewed, "raw_text": draft.raw_text,
            })
        except ValidationError as exc:
            st.error("Review the required fields before saving: " + "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
            ))
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state.saved_receipt = result
            del st.session_state.draft
            st.session_state.pop("receipt_image", None)
            st.rerun()


def tool_evidence(tool: ToolMessage) -> None:
    with st.expander("Receipt evidence"):
        result = tool.artifact
        st.caption(f"{result['start']} to {result['end']}")
        for receipt in result["receipts"]:
            receipt_details(receipt)
        if not result["receipts"]:
            st.write("No matching saved receipts.")


def chat(store: Store) -> None:
    st.subheader("Ask your receipts")
    messages = st.session_state.setdefault("messages", [])
    conversation = st.container(height=400, key="conversation")
    prompt = st.chat_input("Ask about your saved receipts", max_chars=500)
    if prompt:
        messages[:] = messages[-28:] + [{"role": "user", "content": prompt}]
    with conversation:
        if not messages:
            st.caption("Ask about your purchases, food expenses, or where you bought a meal.")
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                for tool in message.get("tools", []):
                    tool_evidence(tool)
                if message.get("error"):
                    st.error(message["error"])
        if prompt:
            response = {"role": "assistant", "content": "", "tools": []}
            try:
                with st.spinner("Checking your receipts…"):
                    history = [{"role": message["role"], "content": message["content"]}
                               for message in messages if not message.get("error")]
                    result = ask_receipts(store, history, datetime.now(UTC).date())
                response["content"] = result["messages"][-1].text
                response["tools"] = [message for message in result["messages"]
                                     if isinstance(message, ToolMessage) and isinstance(message.artifact, dict)]
            except Exception as exc:
                response["error"] = f"The agent could not complete this answer: {exc}"
            messages.append(response)
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Receipt Ledger · Streamlit", page_icon="🧾", layout="wide")
    st.title("Receipt Ledger")
    if not os.getenv("LLM_MODEL"):
        st.info("Set LLM_MODEL and OPENROUTER_API_KEY to enable extraction and chat.")
    store = Store(os.getenv("DB_PATH", "data/receipts.sqlite3"))
    if saved := st.session_state.pop("saved_receipt", None):
        st.success(f"Saved receipt from {saved['merchant']}.")
    upload_tab, chat_tab = st.tabs(["Upload & review", "Ask"])
    with upload_tab:
        upload_and_review(store)
    with chat_tab:
        chat(store)


if __name__ == "__main__":
    main()
