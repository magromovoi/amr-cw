graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "person"
  ]
  node [
    id 1
    label "country"
  ]
  node [
    id 2
    label "and"
  ]
  node [
    id 3
    label "name"
  ]
  node [
    id 4
    label "open"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":mod"
  ]
  edge [
    source 1
    target 3
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 0
    key 0
    label ":op2"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":op2"
  ]
]
