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
    label "name"
  ]
  node [
    id 3
    label "play-01"
  ]
  node [
    id 4
    label "open"
  ]
  node [
    id 5
    label "city"
  ]
  node [
    id 6
    label "event"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":mod"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":name"
  ]
  edge [
    source 1
    target 2
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 4
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
    source 5
    target 2
    key 0
    label ":name"
  ]
  edge [
    source 6
    target 2
    key 0
    label ":name"
  ]
]
