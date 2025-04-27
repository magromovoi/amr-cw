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
    label "and"
  ]
  node [
    id 4
    label "political-party"
  ]
  node [
    id 5
    label "country"
  ]
  node [
    id 6
    label "government-organization"
  ]
  edge [
    source 0
    target 1
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
    source 2
    target 3
    key 0
    label ":ARG1"
  ]
  edge [
    source 4
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 5
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
