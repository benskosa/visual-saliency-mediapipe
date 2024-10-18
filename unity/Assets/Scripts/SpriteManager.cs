using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class SpriteManager : MonoBehaviour
{
    [Serializable] public struct SpriteTag
    {
        public string tag;
        public Sprite sprite;
    }
    [SerializeField] List<SpriteTag> spriteTags = new List<SpriteTag>();
    Dictionary<string, Sprite> spriteTagDict = new Dictionary<string, Sprite>();
    static SpriteManager instance;
    public static SpriteManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<SpriteManager>();
            }
            return instance;
        }
    }
    void Awake()
    {
        if (instance == null)
        {
            instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        foreach (SpriteTag spriteTag in spriteTags)
        {
            spriteTagDict[spriteTag.tag] = spriteTag.sprite;
        }
    }
    Sprite LookUp(string tag)
    {
        if (spriteTagDict.ContainsKey(tag))
        {
            return spriteTagDict[tag];
        }
        else
        {
            // first attempt to find a sprite with the same name as the tag
            // in the folder Icons
            // replace spaces with hyphens
            Sprite sprite = Resources.Load<Sprite>("Icons/" + tag.Replace(' ', '-'));
            if (sprite != null)
            {
                spriteTagDict[tag] = sprite;
                return sprite;
            }
            else
            {
                // if no sprite is found, return null
                return null;
            }
        }
    }
    static public Sprite GetSprite(string tag)
    {
        if (Instance == null)
        {
            return null;
        }
        return Instance.LookUp(tag);
    }
}
