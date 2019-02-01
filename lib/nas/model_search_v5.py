# gumbel softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .operations import OPS, FactorizedReduce, ReLUConvBN
from .genotypes import PRIMITIVES, Genotype
from .construct_utils import random_select, all_select


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      self._ops.append(op)

  def forward(self, x, weights, cpu_weights):
    clist = []
    for j, cpu_weight in enumerate(cpu_weights):
      if abs(cpu_weight) > 1e-10:
        clist.append( weights[j] * self._ops[j](x) )
    assert len(clist) > 0, 'invalid length : {:}'.format(cpu_weights)
    if len(clist) == 1: return clist[0]
    else              : return sum(clist)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    cpu_weights = weights.tolist()
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      clist = []
      if i == 0: indicator = all_select( len(states) )
      else     : indicator = random_select( len(states), 0.6 )

      for j, h in enumerate(states):
        if indicator[j] == 0: continue
        x = self._ops[offset+j](h, weights[offset+j], cpu_weights[offset+j])
        clist.append( x )
      s = sum(clist)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class NetworkV5(nn.Module):

  def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
    super(NetworkV5, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._steps  = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    reduction_prev, cells = False, []
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      cells.append( cell )
      C_prev_prev, C_prev = C_prev, multiplier*C_curr
    self.cells = nn.ModuleList(cells)

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.tau        = 5

    # initialize architecture parameters
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Parameter(torch.Tensor(k, num_ops))
    self.alphas_reduce = Parameter(torch.Tensor(k, num_ops))
    nn.init.normal_(self.alphas_normal, 0, 0.001)
    nn.init.normal_(self.alphas_reduce, 0, 0.001)

  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

  def arch_parameters(self):
    return [self.alphas_normal, self.alphas_reduce]

  def base_parameters(self):
    lists = list(self.stem.parameters()) + list(self.cells.parameters())
    lists += list(self.global_pooling.parameters())
    lists += list(self.classifier.parameters())
    return lists

  def forward(self, inputs):
    batch, C, H, W = inputs.size()
    s0 = s1 = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.gumbel_softmax(self.alphas_reduce, self.tau, True)
      else:
        weights = F.gumbel_softmax(self.alphas_normal, self.tau, True)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    out = out.view(batch, -1)
    logits = self.classifier(out)
    return logits

  def genotype(self):

    def _parse(weights):
      gene, n, start = [], 2, 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j, float(W[j][k_best])))
        start = end
        n += 1
      return gene

    with torch.no_grad():
      gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).cpu().numpy())
      gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).cpu().numpy())

      concat = range(2+self._steps-self._multiplier, self._steps+2)
      genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
      )
    return genotype