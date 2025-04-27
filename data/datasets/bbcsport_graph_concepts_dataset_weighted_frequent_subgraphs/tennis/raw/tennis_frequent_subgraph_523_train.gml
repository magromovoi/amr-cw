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
    label "win-01"
  ]
  node [
    id 3
    label "say-01"
  ]
  node [
    id 4
    label "name"
  ]
  node [
    id 5
    label "open"
  ]
  node [
    id 6
    label "city"
  ]
  node [
    id 7
    label "event"
  ]
  edge [
    source 0
    target 1
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
    target 0
    key 0
    label ":ARG0"
  ]
  edge [
    source 3
    target 0
    key 0
    label ":ARG0"
  ]
  edge [
    source 4
    target 5
    key 0
    label ":op2"
  ]
  edge [
    source 6
    target 4
    key 0
    label ":name"
  ]
  edge [
    source 7
    target 4
    key 0
    label ":name"
  ]
]
