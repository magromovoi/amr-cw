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
    label "manchester"
  ]
  node [
    id 3
    label "say-01"
  ]
  node [
    id 4
    label "team"
  ]
  node [
    id 5
    label "united"
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
    source 1
    target 2
    key 0
    label ":op1"
  ]
  edge [
    source 1
    target 5
    key 0
    label ":op2"
  ]
  edge [
    source 3
    target 0
    key 0
    label ":ARG0"
  ]
  edge [
    source 4
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
