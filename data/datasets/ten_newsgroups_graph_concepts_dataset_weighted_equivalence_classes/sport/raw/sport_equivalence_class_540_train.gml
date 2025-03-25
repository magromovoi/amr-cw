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
    label "year"
  ]
  node [
    id 4
    label "and"
  ]
  node [
    id 5
    label "game"
  ]
  node [
    id 6
    label "city"
  ]
  node [
    id 7
    label "country"
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
    source 2
    target 3
    key 0
    label ":unit"
  ]
  edge [
    source 4
    target 0
    key 0
    label ":op1"
  ]
  edge [
    source 5
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 6
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 7
    target 1
    key 0
    label ":name"
  ]
]
