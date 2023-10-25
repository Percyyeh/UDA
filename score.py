from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np

def whitening(X, method='zca'):
  """
  Whitens the input matrix X using specified whitening method.
  Inputs:
  X:      Input data matrix with data examples along the first dimension
  method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
  'pca_cor', or 'cholesky'.
  """
  X = X.reshape((-1, np.prod(X.shape[1:])))
  X_centered = X - np.mean(X, axis=0)
  Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
  W = None
  
  if method in ['zca', 'pca', 'cholesky']:
    U, Lambda, _ = np.linalg.svd(Sigma)
    if method == 'zca':
      W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
    elif method == 'pca':
      W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
    elif method == 'cholesky':
      W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
      V_sqrt = np.diag(np.std(X, axis=0))
      P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
      G, Theta, _ = np.linalg.svd(P)
      if method == 'zca_cor':
        W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
      elif method == 'pca_cor':
        W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
      raise Exception('Whitening method not found.')
  
  return np.dot(X_centered, W.T)

## either whitening or not
embedding = whitening(embedding, method='cholesky')
embedding = np.concatenate([src_embedding, tgt_embedding], axis=0)

max_k = 100
alpha = 1
            
dpgmm = mixture.BayesianGaussianMixture(
  n_components=args.max_k,
  weight_concentration_prior=args.alpha / args.max_k,
  weight_concentration_prior_type='dirichlet_process',
  covariance_prior=args.embed_dims * np.identity(args.embed_dims),
  covariance_type='full').fit(embedding)

preds = dpgmm.predict(embedding)

for i, (im_target, label_target) in enumerate(target_train_dl):
  im_target = im_target.to(output_device)
  fc1_t = totalNet.feature_extractor.forward(im_target)
  fc1_t, feature_target, fc2_t, predict_prob_target = totalNet.classifier.forward(fc1_t)
  # print('=============', i, feature_target.detach().cpu().numpy().shape)
  tgt_embedding.append(feature_target.detach().cpu().numpy())
  tgt_member.append(label_target[0].detach().cpu().numpy())

src_member = np.concatenate(src_member, axis=0)
tgt_member = np.concatenate(tgt_member, axis=0)

normalized_mutual_info_score(label_target, preds)

