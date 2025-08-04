#!/bin/bash
# User data script for EC2 instances
# Server type: ${server_type}

# Update system
dnf update -y

# Install basic packages
dnf install -y git docker python3 python3-pip

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/ml-app
chown ec2-user:ec2-user /opt/ml-app

# Install Python packages
pip3 install pipenv

# Log completion
echo "User data script completed for ${server_type} server" >> /var/log/user-data.log
