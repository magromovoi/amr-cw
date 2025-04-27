graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "temporal-quantity"
  ]
  node [
    id 1
    label "minute"
  ]
  node [
    id 2
    label "volume-quantity"
  ]
  node [
    id 3
    label "cup"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":unit"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":unit"
  ]
]
