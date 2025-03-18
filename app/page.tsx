"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2 } from "lucide-react"

export default function Home() {
  const [prompt, setPrompt] = useState("")
  const [generatedText, setGeneratedText] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!prompt.trim()) return

    setLoading(true)
    setError("")

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt, length: 200 }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate text")
      }

      setGeneratedText(data.generatedText)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An error occurred while generating text"
      setError(errorMessage)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <div className="w-full max-w-3xl">
        <h1 className="text-3xl font-bold mb-6 text-center">miniGPT Text Generator</h1>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Enter a Prompt</CardTitle>
            <CardDescription>Type a starting text and the model will continue it</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
              <Input
                placeholder="Enter your prompt (e.g., 'The weather today is')"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full"
              />
              <Button type="submit" disabled={loading || !prompt.trim()} className="w-full">
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  "Generate Text"
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {error && <div className="text-red-500 mb-4">{error}</div>}

        {generatedText && (
          <Card>
            <CardHeader>
              <CardTitle>Generated Text</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="whitespace-pre-wrap bg-gray-50 p-4 rounded-md">{generatedText}</div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  )
}

