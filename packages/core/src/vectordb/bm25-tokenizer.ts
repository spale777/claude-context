import winkNLP from 'wink-nlp';
import model from 'wink-eng-lite-web-model';
import * as fs from 'fs';
import * as path from 'path';

/**
 * BM25 tokenizer using wink-nlp for client-side term frequency generation.
 * IDF calculation is handled server-side by Qdrant with the 'idf' modifier.
 * Vocabulary is persisted to ensure consistent sparse vector indices.
 */
export class BM25Tokenizer {
    private nlp: any;
    private vocabulary: Map<string, number>;
    private nextIndex: number;
    private vocabularyPath?: string;
    
    constructor(vocabularyPath?: string) {
        this.nlp = winkNLP(model);
        this.vocabulary = new Map();
        this.nextIndex = 0;
        this.vocabularyPath = vocabularyPath;
        
        if (vocabularyPath) {
            this.loadVocabulary();
        }
    }
    
    /**
     * Tokenize text using wink-nlp
     */
    private tokenize(text: string): string[] {
        const doc = this.nlp.readDoc(text);
        
        // Extract normalized tokens (words only, lowercase)
        return doc.tokens()
            .filter((t: any) => {
                const type = t.out(this.nlp.its.type);
                return type === 'word' && t.out().length >= 2; // Filter out single chars
            })
            .out(this.nlp.its.normal); // Get normalized form
    }
    
    /**
     * Generate sparse vector with term frequencies only.
     * Qdrant will apply IDF modifier server-side.
     */
    generateSparseVector(text: string): { indices: number[]; values: number[] } {
        if (!text || text.trim().length === 0) {
            return { indices: [], values: [] };
        }
        
        const tokens = this.tokenize(text);
        const termFreqs = new Map<string, number>();
        
        // Calculate term frequencies
        for (const token of tokens) {
            termFreqs.set(token, (termFreqs.get(token) || 0) + 1);
        }
        
        if (termFreqs.size === 0) {
            return { indices: [], values: [] };
        }
        
        const indices: number[] = [];
        const values: number[] = [];
        
        // Convert to sparse vector format
        for (const [word, tf] of termFreqs.entries()) {
            // Get or assign vocabulary index
            if (!this.vocabulary.has(word)) {
                this.vocabulary.set(word, this.nextIndex++);
            }
            const index = this.vocabulary.get(word)!;
            
            indices.push(index);
            values.push(tf); // Term frequency only, IDF applied by Qdrant
        }
        
        return { indices, values };
    }
    
    
    /**
     * Load vocabulary from file
     */
    private loadVocabulary(): void {
        if (!this.vocabularyPath || !fs.existsSync(this.vocabularyPath)) {
            return;
        }
        
        try {
            const data = fs.readFileSync(this.vocabularyPath, 'utf-8');
            const vocabData = JSON.parse(data);
            
            this.vocabulary = new Map(Object.entries(vocabData.vocabulary));
            this.nextIndex = vocabData.nextIndex || this.vocabulary.size;
            
            console.log(`[BM25Tokenizer] 📚 Loaded vocabulary: ${this.vocabulary.size} terms`);
        } catch (error) {
            console.warn(`[BM25Tokenizer] ⚠️  Failed to load vocabulary from ${this.vocabularyPath}:`, error);
            this.vocabulary = new Map();
            this.nextIndex = 0;
        }
    }
    
    /**
     * Save vocabulary to file
     */
    saveVocabulary(): void {
        if (!this.vocabularyPath) {
            return;
        }
        
        try {
            // Ensure directory exists
            const dir = path.dirname(this.vocabularyPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            const vocabData = {
                vocabulary: Object.fromEntries(this.vocabulary),
                nextIndex: this.nextIndex,
                timestamp: new Date().toISOString()
            };
            
            fs.writeFileSync(this.vocabularyPath, JSON.stringify(vocabData, null, 2));
            
            console.log(`[BM25Tokenizer] 💾 Saved vocabulary: ${this.vocabulary.size} terms to ${this.vocabularyPath}`);
        } catch (error) {
            console.error(`[BM25Tokenizer] ❌ Failed to save vocabulary to ${this.vocabularyPath}:`, error);
        }
    }
    
}