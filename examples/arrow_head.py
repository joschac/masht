"""Train Masht on the real-world ArrowHead time-series dataset."""

from aeon.datasets import load_arrow_head
from sklearn.metrics import accuracy_score

from masht import Masht


def main() -> None:
    """Load ArrowHead's canonical split, train Masht, and report accuracy."""
    # ArrowHead contains outlines extracted from images of arrowheads. aeon
    # returns its canonical split as (samples, channels, timepoints).
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")

    model = Masht(
        device="auto",
        total_samples=len(X_train) + len(X_test),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"ArrowHead test accuracy: {accuracy_score(y_test, y_pred):.3f}")


if __name__ == "__main__":
    main()
