graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "person"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "temporal-quantity"
  ]
  node [
    id 3
    label "open"
  ]
  node [
    id 4
    label "year"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":age"
  ]
  edge [
    source 1
    target 3
    key 0
    label ":op2"
  ]
  edge [
    source 2
    target 4
    key 0
    label ":unit"
  ]
]
