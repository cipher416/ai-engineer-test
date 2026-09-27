from collections import Counter
from datetime import date
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Index, URL, create_engine, event, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.pool import NullPool

from app.receipts import ReceiptCreate, ReceiptDraft


class Base(DeclarativeBase):
    pass


class Receipt(Base):
    __tablename__ = "receipts"
    __table_args__ = (
        CheckConstraint("total_cents >= 0"), CheckConstraint("tax_cents >= 0"),
        Index("receipt_date", "purchased_on"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    merchant: Mapped[str]
    purchased_on: Mapped[date]
    currency: Mapped[str]
    total_cents: Mapped[int]
    tax_cents: Mapped[int]
    raw_text: Mapped[str]
    warnings: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(server_default=text("CURRENT_TIMESTAMP"))
    items: Mapped[list["Item"]] = relationship(
        cascade="all, delete-orphan", order_by="Item.position", lazy="selectin"
    )

    def as_dict(self) -> dict:
        return {
            **ReceiptDraft.model_validate(self, from_attributes=True).model_dump(mode="json"),
            "id": self.id, "created_at": self.created_at,
        }


class Item(Base):
    __tablename__ = "items"
    __table_args__ = (CheckConstraint("quantity > 0"), CheckConstraint("line_total_cents >= 0"))

    receipt_id: Mapped[str] = mapped_column(ForeignKey("receipts.id", ondelete="CASCADE"), primary_key=True)
    position: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    quantity: Mapped[int]
    line_total_cents: Mapped[int]


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            URL.create("sqlite", database=path), connect_args={"timeout": 10}, poolclass=NullPool
        )

        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON").close()

        with self.engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            Base.metadata.create_all(connection)

    def save(self, receipt: ReceiptCreate | dict) -> dict:
        receipt = ReceiptCreate.model_validate(receipt)
        warnings = []
        if sum(item.line_total_cents for item in receipt.items) + receipt.tax_cents != receipt.total_cents:
            warnings = ["Reviewed total differs from items plus tax (for example, included tax, tips or discounts)."]
        row = Receipt(
            **receipt.model_dump(exclude={"items", "reviewed", "warnings"}), warnings=warnings,
            items=[Item(position=n, **item.model_dump()) for n, item in enumerate(receipt.items)],
        )
        with Session(self.engine) as session, session.begin():
            session.add(row)
            session.flush()
            return row.as_dict()

    def query(self, start: date, end: date, food: str | None = None) -> dict:
        if start > end:
            raise ValueError("start must be on or before end")
        statement = select(Receipt).where(Receipt.purchased_on.between(start, end))
        if food is not None:
            food = food.strip()
            if not food:
                raise ValueError("food must not be empty")
            statement = statement.where(Receipt.items.any(Item.name.icontains(food, autoescape=True)))
        statement = statement.order_by(Receipt.purchased_on, Receipt.id)
        with Session(self.engine) as session:
            receipts = [row.as_dict() for row in session.scalars(statement)]
        totals: Counter[str] = Counter()
        for receipt in receipts:
            totals[receipt["currency"]] += receipt["total_cents"]
        return {
            "start": start.isoformat(), "end": end.isoformat(), "food": food,
            "receipt_count": len(receipts), "totals_cents_by_currency": dict(totals), "receipts": receipts,
        }
