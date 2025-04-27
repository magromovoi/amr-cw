graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "name"
  ]
  node [
    id 1
    label "open"
  ]
  node [
    id 2
    label "australian"
  ]
  node [
    id 3
    label "event"
  ]
  node [
    id 4
    label "country"
  ]
  node [
    id 5
    label "person"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":op2"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 3
    target 0
    key 0
    label ":name"
  ]
  edge [
    source 4
    target 0
    key 0
    label ":name"
  ]
  edge [
    source 5
    target 0
    key 0
    label ":name"
  ]
  edge [
    source 5
    target 4
    key 0
    label ":mod"
  ]
]
