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
    label "company"
  ]
  node [
    id 3
    label "say-01"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 2
    target 1
    key 0
    label ":name"
  ]
  edge [
    source 3
    target 2
    key 0
    label ":ARG0"
  ]
]
