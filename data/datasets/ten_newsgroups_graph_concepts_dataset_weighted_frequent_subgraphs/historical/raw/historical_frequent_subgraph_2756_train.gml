graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "country"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "continent"
  ]
  node [
    id 3
    label "france"
  ]
  node [
    id 4
    label "war"
  ]
  node [
    id 5
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
    target 3
    key 0
    label ":op1"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 4
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 5
    target 0
    key 0
    label ":mod"
  ]
]
