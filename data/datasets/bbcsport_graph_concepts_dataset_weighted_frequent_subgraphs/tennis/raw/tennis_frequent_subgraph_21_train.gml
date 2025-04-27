graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "seed-01"
  ]
  node [
    id 1
    label "person"
  ]
  node [
    id 2
    label "win-01"
  ]
  node [
    id 3
    label "country"
  ]
  node [
    id 4
    label "name"
  ]
  node [
    id 5
    label "ordinal-entity"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":ARG1"
  ]
  edge [
    source 0
    target 5
    key 0
    label ":ord"
  ]
  edge [
    source 1
    target 3
    key 0
    label ":mod"
  ]
  edge [
    source 1
    target 4
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":name"
  ]
]
