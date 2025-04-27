graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "international"
  ]
  node [
    id 1
    label "temporal-quantity"
  ]
  node [
    id 2
    label "1"
  ]
  node [
    id 3
    label "day"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":duration"
  ]
  edge [
    source 1
    target 2
    key 0
    label ":quant"
  ]
  edge [
    source 1
    target 3
    key 0
    label ":unit"
  ]
]
