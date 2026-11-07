import type { ModelId } from '../lib/types'

/**
 * TeX for the equations on the Models page. The words around them live in the dictionaries
 * (models.content.<id>.eq holds one caption per entry here). Every inner array is a list of lines,
 * each typeset as its own display equation, so no alignment environment is needed.
 */
export const MODEL_TEX: Record<ModelId, string[][]> = {
  naive_bayes: [
    ['\\hat y = \\arg\\max_k\\, \\Big[\\, \\log\\, \\pi_k + \\sum_{j=1}^{784} \\log\\, \\mathcal N\\big(x_j;\\, \\mu_{kj},\\, \\sigma_{kj}^2\\big) \\Big]'],
    ['\\log\\, \\mathcal N(x;\\mu,\\sigma^2) = -\\tfrac12 \\log\\,\\big(2\\pi\\sigma^2\\big) - \\frac{(x-\\mu)^2}{2\\sigma^2}'],
    [
      '\\pi_k = \\frac{n_k}{n}, \\qquad \\mu_{kj} = \\frac{1}{n_k}\\sum_{i:\\,y_i=k} x_{ij}',
      '\\sigma_{kj}^2 = \\operatorname{Var}_{i:\\,y_i=k}\\!(x_{ij}) + \\varepsilon\\, \\max_j\\, \\operatorname{Var}_i(x_{ij})',
    ],
  ],
  softmax: [
    ['z = xW + b, \\qquad p_k = \\frac{e^{z_k}}{\\sum_{j=1}^{10} e^{z_j}}'],
    ['L = -\\frac{1}{B}\\sum_{i=1}^{B} \\log\\, p_{i,y_i} + \\frac{\\lambda}{2}\\lVert W \\rVert_F^2'],
    [
      '\\nabla_W L = \\frac{1}{B}\\, X^{\\top}(P - Y) + \\lambda W, \\qquad \\nabla_b L = \\frac{1}{B}\\,\\mathbf 1^{\\top}(P - Y)',
      'W \\leftarrow W - \\eta\\, \\nabla_W L',
    ],
  ],
  mlp: [
    ['a_0 = x, \\qquad z_l = a_{l-1}W_l + b_l, \\qquad a_l = \\frac{\\max(0, z_l) \\odot m_l}{1-p}'],
    [
      '\\delta_L = \\frac{\\operatorname{softmax}(z_L) - Y}{B}',
      '\\delta_{l-1} = (\\delta_l W_l^{\\top}) \\odot \\mathbf 1[a_{l-1} > 0] \\odot m_{l-1}',
      '\\nabla_{W_l} = a_{l-1}^{\\top}\\delta_l',
    ],
    ['v \\leftarrow \\mu v + g, \\qquad \\theta \\leftarrow \\theta - \\eta\\, v'],
  ],
  adaboost: [
    ['G = 1 - \\sum_{k=1}^{K} q_k^2, \\qquad q_k = \\frac{\\sum_{i \\in \\text{node},\\, y_i = k} w_i}{\\sum_{i \\in \\text{node}} w_i}'],
    [
      '\\mathrm{err}_t = \\frac{\\sum_i w_i\\,[h_t(x_i) \\ne y_i]}{\\sum_i w_i}',
      '\\alpha_t = \\eta\\Big(\\ln\\,\\frac{1-\\mathrm{err}_t}{\\mathrm{err}_t} + \\ln(K-1)\\Big)',
      'w_i \\leftarrow w_i\\, e^{\\alpha_t [h_t(x_i) \\ne y_i]}',
    ],
    ['H(x) = \\arg\\max_k \\sum_{t=1}^{T} \\alpha_t\\,[h_t(x) = k]'],
  ],
}
