/* globals customElements */

import { LitElement, html, css } from 'lit-element'
import { fromArrayBuffer } from 'numpy-parser'
import ndarray from 'ndarray'
import Heap from 'heap'

class ImageEmbeddingsKnn extends LitElement {
  static get properties () {
    return {
      queryName: {
        type: String
      },
      results: {
        type: Array
      }
    }
  }

  firstUpdated () {
    (async () => {
      await this.readEmbeddings()
      this.knnByName('image_dandelion_428')
    })()
  }

  async randomKnn () {
    const id = Math.floor(Math.random() * this.embeddings.shape[0])
    this.displayKnn(id)
  }

  async displayKnn (id) {
    const name = this.idToName.get(id)
    console.log(name)
    this.queryName = name

    const results = this.knn(this.embeddings, this.embeddings.pick(id, null), 5)
    this.results = results.map(([id, distance]) => ({ name: this.idToName.get(id), distance }))
  }

  async knnByName (name) {
    this.displayKnn(this.nameToId.get(name))
  }

  async readEmbeddings () {
    const embeddingRaw = await window.fetch('tf_flowers/embeddings/embedding.npy')
    const arrayBuffer = await embeddingRaw.arrayBuffer()
    const { data, shape } = fromArrayBuffer(arrayBuffer)
    this.embeddings = ndarray(data, shape)

    const rawIdName = await window.fetch('tf_flowers/embeddings/id_name.json')
    const idNames = await rawIdName.json()

    this.idToName = new Map(idNames.map(({ id, name }) => ([id, name])))
    this.nameToId = new Map(idNames.map(({ id, name }) => ([name, id])))
  }

  dotProduct (a, b) {
    let dp = 0
    for (let i = 0; i < a.shape[0]; i++) {
      dp += a.get(i) * b.get(i)
    }
    return dp
  }

  knn (embeddings, query, k) {
    // min heap because closest images get the largest dot product
    var heap = new Heap((a, b) => a[1] - b[1])
    for (let i = 0; i < embeddings.shape[0]; i++) {
      heap.push([i, this.dotProduct(query, embeddings.pick(i, null))])
      if (heap.size() > k) {
        heap.pop()
      }
    }
    const results = []
    while (heap.size() > 0) {
      results.push(heap.pop())
    }
    return results.reverse()
  }

  static get styles () {
    return css`
    .wrapper {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      grid-column-gap: 25px;
      grid-row-gap: 25px;
      width:100%;
    }
    .wrapper > div img {
      width:100%;
    }
    `
  }

  render () {
    return html`
    <div style="width:80%;margin-left:auto;margin-right:auto;">
    <div style="font-size:30px;margin-bottom:10px;">Query</div>
    ${this.queryName === undefined ? '' : html`${this.queryName}<br /><img src="tf_flowers/images/${this.queryName}.jpeg" />`}
    <button style="margin-left: 20%; font-size:40px;" @click=${() => this.randomKnn()}>Next random query!</button> <br /> <br />

    <div style="font-size:30px;margin-top: 20px;margin-bottom:10px;">KNN</div>
  <div id="content">
  <div class="wrapper">
    ${this.results === undefined ? '' : this.results.map(({ name, distance }) => html`
    <div>
      ${Math.floor(distance * 100) / 100} ${name} <br />
      <img style="cursor:pointer" @click=${() => this.knnByName(name)} src="tf_flowers/images/${name}.jpeg" />
      </div>`)}
    </div>
  </div>
    `
  }
}

customElements.define('image-embeddings-knn', ImageEmbeddingsKnn)
