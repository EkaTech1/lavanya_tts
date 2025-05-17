#!/bin/bash

# Configuration
VOLUME_MOUNT="/mnt/models"
APP_DIR="/var/www/lavanya_tts"
MODEL_DIR="Fastspeech2_HS"

# Ensure script is run as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

# Create mount point if it doesn't exist
mkdir -p $VOLUME_MOUNT

# Check if volume is already mounted
if ! mountpoint -q $VOLUME_MOUNT; then
    echo "Mounting volume..."
    mount -o discard,defaults,noatime /dev/disk/by-id/scsi-0DO_Volume_tts-models $VOLUME_MOUNT
    if [ $? -ne 0 ]; then
        echo "Failed to mount volume"
        exit 1
    fi
fi

# Ensure mount point has correct permissions
chown -R www-data:www-data $VOLUME_MOUNT
chmod -R 755 $VOLUME_MOUNT

# Copy models if they don't exist on volume
if [ ! -d "$VOLUME_MOUNT/marathi" ] || [ ! -d "$VOLUME_MOUNT/vocoder" ]; then
    echo "Copying model files to volume..."
    cp -r $APP_DIR/$MODEL_DIR/* $VOLUME_MOUNT/
    if [ $? -ne 0 ]; then
        echo "Failed to copy model files"
        exit 1
    fi
fi

# Start/Restart the application
echo "Restarting application..."
systemctl restart lavanya_tts

echo "Deployment completed successfully!" 