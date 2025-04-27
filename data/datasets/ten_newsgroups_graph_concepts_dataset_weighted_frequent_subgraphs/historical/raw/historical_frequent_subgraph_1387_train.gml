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
    label "france"
  ]
  node [
    id 3
    label "war"
  ]
  node [
    id 4
    label "and"
  ]
  node [
    id 5
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
    target 5
    key 0
    label ":mod"
  ]
  edge [
    source 1
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 4
    target 0
    key 0
    label ":op1"
  ]
]
