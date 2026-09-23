import { parseArgs } from 'node:util'
import { preview } from 'vite'

const { values } = parseArgs({ options: {
  host: { type: 'string' },
  port: { type: 'string' },
} })
const overrides = {}
if (values.host) overrides.host = values.host
if (values.port) {
  const port = Number(values.port)
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid port')
  overrides.port = port
}
const server = await preview({ preview: overrides })
server.printUrls()
