# docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg golang-go
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo groupadd docker
sudo usermod -aG docker $USER

# kind
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
# sudo kind create cluster

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
echo "source <(kubectl completion bash)" >> ~/.bashrc
echo "alias k=kubectl" >> ~/.bashrc
echo "complete -o default -F __start_kubectl k" >> ~/.bashrc
source ~/.bashrc

# kn
wget https://github.com/knative/client/releases/download/knative-v1.11.2/kn-linux-amd64
mv kn-linux-amd64 kn
chmod +x kn
sudo mv kn /usr/local/bin
wget https://github.com/knative-extensions/kn-plugin-quickstart/releases/download/knative-v1.11.1/kn-quickstart-linux-amd64
mv kn-quickstart-linux-amd64 kn-quickstart
chmod +x kn-quickstart
sudo mv kn-quickstart /usr/local/bin

sudo kn quickstart kind
sudo kind get clusters
sudo chown $(id -u):$(id -g) $HOME/.kube/config

curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
sudo mv cosign-linux-amd64 /usr/local/bin/cosign
sudo chmod +x /usr/local/bin/cosign
wget https://github.com/jqlang/jq/releases/download/jq-1.7/jq-linux-amd64
mv jq-linux-amd64 jq
chmod +x jq
sudo mv jq /usr/local/bin/jq

curl -sSL https://github.com/knative/serving/releases/download/knative-v1.10.1/serving-core.yaml \
  | grep 'gcr.io/' | awk '{print $2}' | sort | uniq \
  | xargs -n 1 \
    cosign verify -o text \
      --certificate-identity=signer@knative-releases.iam.gserviceaccount.com \
      --certificate-oidc-issuer=https://accounts.google.com

kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-core.yaml

# istio
# curl -L https://istio.io/downloadIstio | sh -
# cd istio*
# chmod +x ./bin/istioctl
# sudo mv ./bin/istioctl /usr/local/bin/istioctl
# cd ~
# istioctl install -y
kubectl apply -l knative.dev/crd-install=true -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/istio.yaml
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/net-istio.yaml
kubectl get pods -n knative-serving

# Cert
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml
OS=$(go env GOOS); ARCH=$(go env GOARCH); curl -fsSL -o cmctl.tar.gz https://github.com/cert-manager/cert-manager/releases/latest/download/cmctl-$OS-$ARCH.tar.gz
tar xzf cmctl.tar.gz
sudo mv cmctl /usr/local/bin
while
  READY=$(cmctl check api)
  echo $READY
  sleep 5
  [ "$READY" != "The cert-manager API is ready" ]
do :; done

# KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-runtimes.yaml

# DNS patch
kubectl patch configmap/config-domain \
      --namespace knative-serving \
      --type merge \
      --patch '{"data":{"example.com":""}}'

# Create cluster
sudo kubeadm init --image-repository registry.aliyuncs.com/google_containers --pod-network-cidr 10.244.0.0/16 --cri-socket unix:///var/run/crio/crio.sock