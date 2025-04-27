graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "temporal-quantity"
  ]
  node [
    id 1
    label "1"
  ]
  node [
    id 2
    label "day"
  ]
  node [
    id 3
    label "ordinal-entity"
  ]
  node [
    id 4
    label "team"
  ]
  node [
    id 5
    label "name"
  ]
  node [
    id 6
    label "country"
  ]
  node [
    id 7
    label "person"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":quant"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":unit"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":value"
  ]
  edge [
    source 4
    target 5
    key 0
    label ":name"
  ]
  edge [
    source 6
    target 5
    key 0
    label ":name"
  ]
  edge [
    source 7
    target 5
    key 0
    label ":name"
  ]
]
