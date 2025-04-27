graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "name"
  ]
  node [
    id 1
    label "war"
  ]
  node [
    id 2
    label "country"
  ]
  node [
    id 3
    label "and"
  ]
  node [
    id 4
    label "person"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":op2"
  ]
  edge [
    source 2
    target 0
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":op1"
  ]
]
