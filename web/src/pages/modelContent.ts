import type { ModelKind } from '../engine/contract'

/**
 * TeX for the equations on the Models page. The words around them live in the dictionaries
 * (models.content.<id>.eq holds one caption per entry here). Every inner array is a list of lines,
 * each typeset as its own display equation, so no alignment environment is needed.
 */
/** Architectures with a full article (texts in models.content.<kind>); the linear SVM uses the compact one. */
export const CONTENT_KINDS = ['naive_bayes', 'softmax', 'mlp', 'cnn', 'adaboost'] as const
export type ContentKind = (typeof CONTENT_KINDS)[number]
export const hasContent = (k: ModelKind): k is ContentKind => (CONTENT_KINDS as readonly string[]).includes(k)

export const MODEL_TEX: Record<ContentKind, string[][]> = {
  naive_bayes: [
    ['\\hat y = \\operatorname*{arg\\,max}_k \\Big[\\, \\log \\pi_k + \\sum_{j=1}^{784} \\log \\mathcal N\\big(x_j;\\, \\mu_{kj},\\, \\sigma_{kj}^2\\big) \\Big]'],
    ['\\log \\mathcal N(x;\\mu,\\sigma^2) = -\\tfrac12 \\log \\big(2\\pi\\sigma^2\\big) - \\frac{(x-\\mu)^2}{2\\sigma^2}'],
    [
      '\\pi_k = \\frac{n_k}{n}, \\qquad \\mu_{kj} = \\frac{1}{n_k}\\sum_{i:\\,y_i=k} x_{ij}',
      '\\sigma_{kj}^2 = \\operatorname{Var}_{i:\\,y_i=k}(x_{ij}) + \\varepsilon\\, \\max_j \\operatorname{Var}_i(x_{ij})',
    ],
  ],
  softmax: [
    ['z = xW + b, \\qquad p_k = \\frac{e^{z_k}}{\\sum_{j=1}^{10} e^{z_j}}'],
    ['L = -\\frac{1}{B}\\sum_{i=1}^{B} \\log p_{i,y_i} + \\frac{\\lambda}{2}\\lVert W \\rVert_F^2'],
    [
      '\\nabla_W L = \\frac{1}{B}\\, X^{\\top}(P - Y) + \\lambda W, \\qquad \\nabla_b L = \\frac{1}{B}\\,\\mathbf{1}^{\\top}(P - Y)',
      'W \\leftarrow W - \\eta\\, \\nabla_W L',
    ],
  ],
  mlp: [
    ['a_0 = x, \\qquad z_l = a_{l-1}W_l + b_l, \\qquad a_l = \\frac{\\max(0, z_l) \\odot m_l}{1-p}'],
    [
      '\\delta_L = \\frac{\\operatorname{softmax}(z_L) - Y}{B}',
      '\\delta_{l-1} = (\\delta_l W_l^{\\top}) \\odot \\mathbf{1}[a_{l-1} > 0] \\odot m_{l-1}',
      '\\nabla_{W_l} = a_{l-1}^{\\top}\\delta_l',
    ],
    ['v \\leftarrow \\mu v + g, \\qquad \\theta \\leftarrow \\theta - \\eta\\, v'],
  ],
  cnn: [
    ['z_{o,i,j} = b_o + \\sum_{c=1}^{C_{\\text{in}}} \\sum_{u=0}^{k-1} \\sum_{v=0}^{k-1} K_{o,c,u,v}\\; x_{c,\\, i+u,\\, j+v}'],
    ['a = \\max(0, z), \\qquad p_{c,i,j} = \\max_{0 \\le u,\\, v < s}\\; a_{c,\\, s i+u,\\, s j+v}'],
    ['h = \\operatorname{vec}(p), \\qquad o = hW + b, \\qquad \\hat y = \\operatorname{softmax}(o)'],
    [
      'P = \\operatorname{im2col}(x), \\qquad Z = P\\, K_{\\text{mat}}^{\\top} + \\mathbf{1} b^{\\top}',
      '\\nabla_{K_{\\text{mat}}} = \\Delta^{\\top} P, \\qquad \\nabla_b = \\Delta^{\\top} \\mathbf{1}, \\qquad \\nabla_x = \\operatorname{col2im}\\big(\\Delta\\, K_{\\text{mat}}\\big)',
      '\\frac{\\partial L}{\\partial a_{c,\\,si+u,\\,sj+v}} = \\frac{\\partial L}{\\partial p_{c,i,j}}\\; \\mathbf{1}\\big[(u, v) = \\operatorname{arg\\,max}\\big]',
    ],
  ],
  adaboost: [
    ['G = 1 - \\sum_{k=1}^{K} q_k^2, \\qquad q_k = \\frac{\\sum_{i \\in \\text{node},\\, y_i = k} w_i}{\\sum_{i \\in \\text{node}} w_i}'],
    [
      '\\mathrm{err}_t = \\frac{\\sum_i w_i\\,[h_t(x_i) \\ne y_i]}{\\sum_i w_i}',
      '\\alpha_t = \\eta\\Big(\\ln \\frac{1-\\mathrm{err}_t}{\\mathrm{err}_t} + \\ln(K-1)\\Big)',
      'w_i \\leftarrow w_i\\, e^{\\alpha_t [h_t(x_i) \\ne y_i]}',
    ],
    ['H(x) = \\operatorname*{arg\\,max}_k \\sum_{t=1}^{T} \\alpha_t\\,[h_t(x) = k]'],
  ],
}

/** Linear SVM and stacking: measured in Python only (results pages), captions in models.extra.<id>.eq. */
export const EXTRA_TEX: Record<'svm' | 'stacking', string[][]> = {
  svm: [
    ['z = xW + b, \\qquad \\hat y = \\operatorname*{arg\\,max}_k z_k'],
    ['L = \\frac{1}{B}\\sum_{i=1}^{B} \\sum_{k} \\max\\big(0,\\, 1 - t_{ik} z_{ik}\\big) + \\frac{\\lambda}{2}\\lVert W \\rVert_F^2, \\qquad t_{ik} = \\pm 1'],
  ],
  stacking: [
    ['f(x) = \\big[\\log \\max\\big(p^{(1)}(x), 10^{-4}\\big),\\, \\dots,\\, z^{\\text{SVM}}(x)\\big]'],
    ['\\hat y = \\operatorname*{arg\\,max}_k \\operatorname{softmax}\\big(\\tilde f(x)\\, W + b\\big)_k'],
  ],
}
