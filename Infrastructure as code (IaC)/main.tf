# Terraform Configuration for ML Churn Prediction Project
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Variables for easy customization
variable "customer churn predict" {
  description = "Name for churn prediction project"
  type        = string
  default     = "churn-ml"
}

# Provider configuration - us-east-1 for free tier benefits
provider "aws" {
  region = "us-east-1"
}

# Random suffix to make bucket names unique
resource "random_id" "suffix" {
  byte_length = 4
}

# Common tags for all resources
locals {
  common_tags = {
    Project = var.project_name
    Purpose = "Customer churn prediction ML pipeline"
  }
}

# VPC - Virtual Private Cloud for ML infrastructure
resource "aws_vpc" "ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-vpc"
  })
}

# Public subnet for EC2 instances
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.ml_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "us-east-1a"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-public-subnet"
  })
}

# Private subnet for RDS database
resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-1b"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-private-subnet"
  })
}

# Internet Gateway for public access
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.ml_vpc.id

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-igw"
  })
}

# Route table for internet access
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ml_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-public-routes"
  })
}

# Associate route table with public subnet
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# Security Group for ML servers (Streamlit, FastAPI, MLflow, Prefect)
resource "aws_security_group" "ml_servers" {
  name        = "${var.project_name}-servers-sg"
  description = "Security group for ML application servers"
  vpc_id      = aws_vpc.ml_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP/HTTPS
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # ML application ports
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "MLflow server"
  }

  ingress {
    from_port   = 8000
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "FastAPI and Streamlit"
  }

  ingress {
    from_port   = 4200
    to_port     = 4200
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Prefect server"
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-servers-sg"
  })
}

# Security Group for PostgreSQL database
resource "aws_security_group" "database" {
  name        = "${var.project_name}-db-sg"
  description = "Security group for PostgreSQL database"
  vpc_id      = aws_vpc.ml_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ml_servers.id]
    description     = "PostgreSQL from ML servers"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-database-sg"
  })
}

# Random secure password for database
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# IAM role for EC2 to access S3 (for MLflow artifacts)
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# Attach S3 access policy to role
resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

# Instance profile for EC2
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# Latest Amazon Linux 2023 AMI (free tier eligible)
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

# EC2 Instances for ML services (free tier t2.micro)
resource "aws_instance" "ml_servers" {
  for_each = {
    streamlit = "Streamlit web app server"
    fastapi   = "FastAPI prediction server"
    mlflow    = "MLflow experiment tracking"
    prefect   = "Prefect workflow management"
  }

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t2.micro" # Free tier eligible
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ml_servers.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  # Basic setup script
  user_data = base64encode(<<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker git python3 python3-pip
    systemctl start docker
    systemctl enable docker
    usermod -a -G docker ec2-user
    pip3 install pipenv
    echo "${each.value} server setup complete" > /home/ec2-user/server_info.txt
  EOF
  )

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${each.key}"
    Type = each.value
  })
}

# PostgreSQL database for MLflow (free tier eligible)
resource "aws_db_instance" "mlflow_db" {
  allocated_storage      = 20 # Free tier limit
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.micro" # Free tier eligible
  db_name                = "mlflow"
  username               = "mlflow_user"
  password               = random_password.db_password.result
  skip_final_snapshot    = true
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-mlflow-db"
  })
}

# Database subnet group (required for RDS in VPC)
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnets"
  subnet_ids = [aws_subnet.public.id, aws_subnet.private.id]

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-db-subnet-group"
  })
}

# S3 bucket for ML artifacts and models
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-artifacts-${random_id.suffix.hex}"

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-artifacts"
  })
}

# S3 bucket security settings
resource "aws_s3_bucket_public_access_block" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Outputs - Important information after deployment
output "server_ips" {
  description = "Public IP addresses of ML servers"
  value = {
    for name, instance in aws_instance.ml_servers : name => instance.public_ip
  }
}

output "database_endpoint" {
  description = "PostgreSQL database endpoint"
  value       = aws_db_instance.mlflow_db.endpoint
  sensitive   = true
}

output "database_password" {
  description = "Database password (keep secure)"
  value       = random_password.db_password.result
  sensitive   = true
}

output "s3_bucket" {
  description = "S3 bucket name for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "connection_info" {
  description = "How to connect to your servers"
  value = {
    streamlit = "http://${aws_instance.ml_servers["streamlit"].public_ip}:8501"
    fastapi   = "http://${aws_instance.ml_servers["fastapi"].public_ip}:8000"
    mlflow    = "http://${aws_instance.ml_servers["mlflow"].public_ip}:5000"
    prefect   = "http://${aws_instance.ml_servers["prefect"].public_ip}:4200"
  }
}
