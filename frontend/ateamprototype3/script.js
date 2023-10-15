const client = ZoomMtgEmbedded.createClient()

let meetingSDKElement = document.getElementById('meetingSDKElement')

client.init({ zoomAppRoot: meetingSDKElement, language: 'en-US' })

client.join({
  sdkKey: 'nP4aJBaSkGzCQ8V35rb8g',
  signature: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZGtLZXkiOiJuUDRhSkJhU2tHekNROFYzNXJiOGciLCJhcHBLZXkiOiJuUDRhSkJhU2tHekNROFYzNXJiOGciLCJtbiI6NzQ4MTQ5NjU4MDQsInJvbGUiOjAsImlhdCI6MTY5NzM0NTgzMywiZXhwIjoxNjk3MzUzMDMzLCJ0b2tlbkV4cCI6MTY5NzM1MzAzM30.KdOBQ-U-Hui0vgD1ORa6xC86occxYGh-vxyuXY2rtes', // role in SDK signature needs to be 0
  meetingNumber: '74814965804',
  password: 'ZNqMoRJufbmtKb30r6Tnee7v7xioDD.1',
  userName: 'ateam_bot'
})