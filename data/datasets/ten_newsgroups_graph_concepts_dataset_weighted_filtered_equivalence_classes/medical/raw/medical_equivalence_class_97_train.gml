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
    label "write-01"
  ]
  node [
    id 3
    label "article"
  ]
  node [
    id 4
    label "date-entity"
  ]
  node [
    id 5
    label "1993"
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
    label ":medium"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":time"
  ]
  edge [
    source 4
    target 5
    key 0
    label ":year"
  ]
]
