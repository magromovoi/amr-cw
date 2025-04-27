graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "disease"
  ]
  node [
    id 1
    label "name"
  ]
  node [
    id 2
    label "have-03"
  ]
  node [
    id 3
    label "person"
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
    label ":ARG1"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":ARG0"
  ]
]
