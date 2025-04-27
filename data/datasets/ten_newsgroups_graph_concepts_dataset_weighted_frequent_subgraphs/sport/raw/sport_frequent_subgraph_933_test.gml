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
    label "say-01"
  ]
  node [
    id 3
    label "games"
  ]
  node [
    id 4
    label "game"
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
    label ":op2"
  ]
  edge [
    source 2
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
]
