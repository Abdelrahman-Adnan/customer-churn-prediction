# ML Churn Prediction Infrastructure 🚀

Simple Terraform setup for deploying a complete machine learning pipeline on AWS.

## What's This? 🤔

This Terraform file creates everything you need to run your customer churn prediction models in the cloud:

- **4 EC2 servers** (Streamlit, FastAPI, MLflow, Prefect)
- **PostgreSQL database** for MLflow experiments
- **S3 bucket** for storing models and artifacts
- **VPC networking** to connect everything securely

## Quick Start 🏃‍♂️

```bash
# Setup
terraform init
terraform plan

# Deploy
terraform apply

# Clean up when done
terraform destroy
```

## What Gets Created 📦

| Service | Purpose | Port |
|---------|---------|------|
| 🎨 Streamlit | Web dashboard | 8501 |
| ⚡ FastAPI | Prediction API | 8000 |
| 📊 MLflow | Experiment tracking | 5000 |
| 🔄 Prefect | Workflow automation | 4200 |
| 🐘 PostgreSQL | Database | 5432 |
| 📁 S3 Bucket | File storage | - |

## Costs 💰

Everything uses AWS free tier:
- EC2: t2.micro instances (750 hours/month free)
- RDS: db.t3.micro with 20GB storage
- S3: 5GB storage free
- VPC: No additional charges

**Monthly cost**: ~$0 for first year with free tier

## After Deployment 🎉

You'll get output with:
- Server IP addresses
- Database connection details
- Direct links to each service

Example:
```
Streamlit: http://34.123.45.67:8501
FastAPI: http://34.123.45.68:8000
MLflow: http://34.123.45.69:5000
```

## Customization ⚙️

Change the project name:
```hcl
variable "project_name" {
  default = "your-project-name"
}
```

## Security Notes 🔒

- Database only accessible from ML servers
- S3 bucket blocks public access
- Security groups limit port access
- SSH access available on port 22

## Troubleshooting 🔧

**Issue**: Terraform fails with provider errors
**Fix**: Run `terraform init` first

**Issue**: Resources already exist
**Fix**: Check if you have existing resources with same names

**Issue**: Permission denied
**Fix**: Configure AWS credentials: `aws configure`

## File Structure 📁

```
├── main.tf              # Main infrastructure code
├── terraform.tfvars     # Your custom variables (optional)
└── README.md            # This file
```
