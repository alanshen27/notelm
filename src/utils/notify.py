import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def send_email(subject: str, body: str) -> bool:
    _load_dotenv()

    to = os.environ.get("NOTIFY_EMAIL")
    if not to:
        print("NOTIFY_EMAIL not set — skipping email notification")
        return False

    password = os.environ.get("SMTP_PASS") or os.environ.get("SMTP_PASSWORD")
    if not password:
        print("SMTP_PASS not set — skipping email notification")
        return False

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", to)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)

    print(f"Notification sent to {to}")
    return True


def notify_training_complete(
    *,
    success: bool,
    epochs: int,
    device: str,
    elapsed_s: float,
    weights_path: str = "weights.pt",
    error: str | None = None,
):
    status = "finished" if success else "failed"
    subject = f"notelm training {status}"

    lines = [
        f"Training {status}.",
        "",
        f"Epochs:  {epochs}",
        f"Device:  {device}",
        f"Elapsed: {elapsed_s / 3600:.2f} h ({elapsed_s:.0f} s)",
    ]
    if success:
        lines.append(f"Weights: {weights_path}")
    if error:
        lines.extend(["", "Error:", error])

    lines.extend(["", f"Sent at {datetime.now().isoformat(timespec='seconds')}"])
    send_email(subject, "\n".join(lines))
