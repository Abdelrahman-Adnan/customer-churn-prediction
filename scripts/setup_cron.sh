#!/bin/bash

# Cron Job Setup Script for MLOps Customer Churn Prediction
# =========================================================

set -e

PROJECT_DIR=$(pwd)
LOG_DIR="$PROJECT_DIR/logs"

echo "Setting up automated cron jobs for MLOps pipeline..."
echo "Project directory: $PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Backup existing crontab
echo "Backing up existing crontab..."
crontab -l > "$LOG_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "No existing crontab found"

# Create temporary crontab file
TEMP_CRON=$(mktemp)

# Add existing crontab entries (if any)
crontab -l 2>/dev/null >> "$TEMP_CRON" || true

# Add MLOps cron jobs
echo "" >> "$TEMP_CRON"
echo "# MLOps Customer Churn Prediction - Automated Jobs" >> "$TEMP_CRON"
echo "# Generated on $(date)" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Training job: Every 3 days at 9 AM
echo "# Automated model training - Every 3 days at 9 AM" >> "$TEMP_CRON"
echo "0 9 */3 * * cd $PROJECT_DIR && docker-compose run --rm training python services/training/churn_mlops_pipeline.py >> $LOG_DIR/training_cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Monitoring job: Every hour
echo "# Automated monitoring - Every hour" >> "$TEMP_CRON"
echo "0 * * * * cd $PROJECT_DIR && docker-compose run --rm monitoring python services/monitoring/monitor_churn_model.py >> $LOG_DIR/monitoring_cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Install the new crontab
echo "Installing new crontab..."
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

# Create log files if they don't exist
touch "$LOG_DIR/training_cron.log"
touch "$LOG_DIR/monitoring_cron.log"

echo "✅ Cron jobs configured successfully!"
echo ""
echo "📋 Scheduled Jobs:"
echo "  🔄 Training:   Every 3 days at 9:00 AM"
echo "  📊 Monitoring: Every hour"
echo ""
echo "📝 Log files:"
echo "  Training:   $LOG_DIR/training_cron.log"
echo "  Monitoring: $LOG_DIR/monitoring_cron.log"
echo ""
echo "🔍 View active cron jobs:"
echo "  crontab -l"
echo ""
echo "📊 Monitor logs:"
echo "  tail -f $LOG_DIR/training_cron.log"
echo "  tail -f $LOG_DIR/monitoring_cron.log"
echo ""
echo "⚠️  Note: Make sure Docker and docker-compose are installed and accessible"
echo "⚠️  Note: Ensure the project containers are built: make build"
