graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "continent"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "europe"
  ]
  node [
    id 3
    label "britain"
  ]
  node [
    id 4
    label "world-region"
  ]
  node [
    id 5
    label "country"
  ]
  node [
    id 6
    label "person"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 1
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 1
    target 3
    key 0
    label ":op1"
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
    target 5
    key 0
    label ":mod"
  ]
]
