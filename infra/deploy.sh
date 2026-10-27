#!/usr/bin/env bash
# Usage: REPO_URL=https://github.com/<you>/vibrovision.git ./infra/deploy.sh [stack-name] [ami-id]
set -euo pipefail
STACK=${1:-vibrovision}
AMI=${2:-}
: "${REPO_URL:?set REPO_URL to the git URL of this repository}"
MYIP=$(curl -s https://checkip.amazonaws.com)/32
aws cloudformation deploy --stack-name "$STACK" --template-file "$(dirname "$0")/graviton-ec2.yaml" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides RepoUrl="$REPO_URL" AmiId="$AMI" AllowedCidr="${ALLOWED_CIDR:-$MYIP}" \
      ApiKey="${API_KEY:-}" AlertEmail="${ALERT_EMAIL:-}"
aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
echo "First boot builds the container (~5 min). Then: curl <Endpoint>healthz"
