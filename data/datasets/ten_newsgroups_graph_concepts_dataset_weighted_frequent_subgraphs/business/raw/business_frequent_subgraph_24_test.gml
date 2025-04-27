graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "have-org-role-91"
  ]
  node [
    id 1
    label "company"
  ]
  node [
    id 2
    label "person"
  ]
  node [
    id 3
    label "analyze-01"
  ]
  node [
    id 4
    label "name"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":ARG1"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":ARG0"
  ]
  edge [
    source 2
    target 4
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
