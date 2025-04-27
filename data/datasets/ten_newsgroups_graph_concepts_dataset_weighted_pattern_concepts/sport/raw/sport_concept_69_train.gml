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
    label "game"
  ]
  node [
    id 5
    label "city"
  ]
  node [
    id 6
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
    target 1
    key 0
    label ":name"
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
]
