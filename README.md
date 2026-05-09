# Emiya Live Trader

Diffusion-model paper trading bot for SPY using Alpaca market/trading APIs and FRED 10Y Treasury yield data.

## Secret Handling

The code reads API keys from environment variables:

```bash
ALPACA_API_KEY
ALPACA_SECRET_KEY
FRED_API_KEY
```

Do not put real keys in GitHub. Use `.env.example` only as a template. On Vultr, store real keys in `/etc/emiya/live-trader.env` with root-only permissions.

## Required Runtime Files

For live trading, the server needs:

```text
live_trader.py
config.py
models/
trading/
training/
data/
diffusion_macro_v2.pth
diffusion_macro_v2_scaler.pkl
macro_cache.csv
```

Training CSVs are not required for live mode unless you plan to retrain on the server.

## Vultr Deployment Through GitHub

These steps assume a fresh Ubuntu/Debian Vultr instance and a GitHub repo containing this project.

### 1. Create a Server User

```bash
sudo adduser --disabled-password --gecos "" emiya
sudo mkdir -p /opt/emiya
sudo chown emiya:emiya /opt/emiya
```

### 2. Install System Packages

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

### 3. Clone From GitHub

For a public repo:

```bash
sudo -u emiya git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /opt/emiya/Emiya1
```

For a private repo, use a deploy key or GitHub SSH key, then clone:

```bash
sudo -u emiya git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git /opt/emiya/Emiya1
```

### 4. Install Python Dependencies

```bash
cd /opt/emiya/Emiya1
sudo -u emiya python3 -m venv .venv
sudo -u emiya .venv/bin/pip install --upgrade pip
sudo -u emiya .venv/bin/pip install -r requirements-torch-cpu.txt
sudo -u emiya .venv/bin/pip install -r requirements.txt
```

On a 1 vCPU instance, run live inference only. Train the model elsewhere and push/copy the `.pth` and scaler files.

### 5. Add Secrets On The Server

```bash
sudo mkdir -p /etc/emiya
sudo nano /etc/emiya/live-trader.env
```

Put this in the file:

```bash
ALPACA_API_KEY=your_real_key
ALPACA_SECRET_KEY=your_real_secret
FRED_API_KEY=your_real_fred_key
```

Lock it down:

```bash
sudo chown root:root /etc/emiya/live-trader.env
sudo chmod 600 /etc/emiya/live-trader.env
```

### 6. Install The systemd Service

```bash
sudo cp /opt/emiya/Emiya1/deploy/emiya-live-trader.service /etc/systemd/system/emiya-live-trader.service
sudo systemctl daemon-reload
sudo systemctl enable emiya-live-trader
sudo systemctl start emiya-live-trader
```

### 7. Check Logs

```bash
sudo systemctl status emiya-live-trader
sudo journalctl -u emiya-live-trader -f
```

### 8. Update From GitHub

```bash
cd /opt/emiya/Emiya1
sudo -u emiya git pull
sudo -u emiya .venv/bin/pip install -r requirements-torch-cpu.txt
sudo -u emiya .venv/bin/pip install -r requirements.txt
sudo systemctl restart emiya-live-trader
```

## Local Smoke Test

Before enabling the service, you can test imports and model loading:

```bash
python -m compileall .
python main.py --no-train --branches 2 --horizon 3 --equity 10000 --device cpu
```

## Security Notes

- Keep the GitHub repo private if it contains model checkpoints or trading research you do not want public.
- Never commit `.env`, API keys, shell history containing keys, or screenshots of credentials.
- If a key was ever committed, rotate it immediately in Alpaca/FRED.
- Alpaca is currently configured for paper trading in `trading/alpaca_client.py`.
"# APStatsTrader" 
