#!/bin/bash
# Setup cron job for daily trading
# Run this script once on your cloud VM

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up cron job for automated trading..."

# Make run script executable
chmod +x "${PROJECT_DIR}/scripts/run_scheduled.sh"

# Create cron entry
# Run Monday-Friday at 9:50 AM ET (14:50 UTC, adjust for your timezone)
# Format: minute hour day month day-of-week command
CRON_SCHEDULE="50 14 * * 1-5"  # 9:50 AM ET = 14:50 UTC
CRON_COMMAND="${PROJECT_DIR}/scripts/run_scheduled.sh"
CRON_ENTRY="${CRON_SCHEDULE} ${CRON_COMMAND}"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "${CRON_COMMAND}"; then
    echo "Cron job already exists. Skipping..."
else
    # Add cron job
    (crontab -l 2>/dev/null || echo "") | { cat; echo "${CRON_ENTRY}"; } | crontab -
    echo "Cron job added successfully!"
fi

# Display current crontab
echo ""
echo "Current cron schedule:"
crontab -l

echo ""
echo "Setup complete!"
echo "Trading will run Monday-Friday at 9:50 AM ET (14:50 UTC)"
echo ""
echo "To view logs:"
echo "  tail -f ${PROJECT_DIR}/logs/live/trading_*.log"
echo ""
echo "To test manually:"
echo "  ${CRON_COMMAND}"
