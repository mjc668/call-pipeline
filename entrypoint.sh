#!/bin/bash
set -e

echo "0 1 * * * root python /app/reporting/daily_summary.py --date yesterday >> /var/log/cron.log 2>&1" > /etc/cron.d/call-pipeline
echo "0 0 1 * * root python /app/reporting/monthly_report.py >> /var/log/cron.log 2>&1" >> /etc/cron.d/call-pipeline
chmod 0644 /etc/cron.d/call-pipeline

cron

exec "$@"
