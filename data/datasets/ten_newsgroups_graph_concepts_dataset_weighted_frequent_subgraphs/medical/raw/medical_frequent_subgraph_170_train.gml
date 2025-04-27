graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "write-01"
  ]
  node [
    id 1
    label "person"
  ]
  node [
    id 2
    label "article"
  ]
  node [
    id 3
    label "date-entity"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":medium"
  ]
  edge [
    source 2
    target 3
    key 0
    label ":time"
  ]
]
