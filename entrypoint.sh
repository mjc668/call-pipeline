#!/bin/bash
set -e

TZ=$(python3 -c "
import yaml
with open('/config/config.yaml') as f:
    c = yaml.safe_load(f)
print(c.get('timezone', 'UTC'))
" 2>/dev/null || echo "UTC")
export TZ
echo "Timezone: $TZ"

echo "0 1 * * * root TZ=$TZ python /app/reporting/daily_summary.py --date yesterday >> /var/log/cron.log 2>&1" > /etc/cron.d/call-pipeline
echo "0 0 1 * * root TZ=$TZ python /app/reporting/monthly_report.py >> /var/log/cron.log 2>&1" >> /etc/cron.d/call-pipeline
chmod 0644 /etc/cron.d/call-pipeline

cron

exec "$@"
