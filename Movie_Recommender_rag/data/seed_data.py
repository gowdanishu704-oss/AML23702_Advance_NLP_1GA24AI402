"""
Seed data definitions and loader for Neo4j Movies Knowledge Graph.
Contains the canonical Neo4j Movies dataset with Movies, Persons, and Relationships.
"""

from typing import List, Dict, Any

MOVIES: List[Dict[str, Any]] = [
    {"title": "The Matrix", "released": 1999, "tagline": "Welcome to the Real World"},
    {"title": "The Matrix Reloaded", "released": 2003, "tagline": "Free your mind"},
    {"title": "The Matrix Revolutions", "released": 2003, "tagline": "Everything that has a beginning has an end"},
    {"title": "The Devil's Advocate", "released": 1997, "tagline": "Evil has its winning ways"},
    {"title": "A Few Good Men", "released": 1992, "tagline": "In the heart of the nation's capital, in a courthouse of the U.S. government, one man will stop at nothing to keep his honor, and one will stop at nothing to find the truth."},
    {"title": "Top Gun", "released": 1986, "tagline": "I feel the need, the need for speed."},
    {"title": "Jerry Maguire", "released": 2000, "tagline": "The rest of his life begins now."},
    {"title": "Stand By Me", "released": 1986, "tagline": "For some, it's the last real adventure of a childhood, for a kid it's a lifetime."},
    {"title": "As Good as It Gets", "released": 1997, "tagline": "A comedy from the heart that goes for the throat."},
    {"title": "What Dreams May Come", "released": 1998, "tagline": "After life there is more. The end is just the beginning."},
    {"title": "Snow Falling on Cedars", "released": 1999, "tagline": "First love lasts forever."},
    {"title": "You've Got Mail", "released": 1998, "tagline": "At snacks and coffee, two enemies fell in love."},
    {"title": "Sleepless in Seattle", "released": 1993, "tagline": "What if someone you never met, someone you never saw, someone you never knew was the only someone for you?"},
    {"title": "Joe Versus the Volcano", "released": 1990, "tagline": "A story of love, lava and living."},
    {"title": "When Harry Met Sally", "released": 1989, "tagline": "Can two friends sleep together and still remain friends?"},
    {"title": "That Thing You Do", "released": 1996, "tagline": "In every life there comes a time when that thing you dream becomes that thing you do."},
    {"title": "The Replacements", "released": 2000, "tagline": "Pain heals, Chicks dig scars... Glory lasts forever"},
    {"title": "Rescue Dawn", "released": 2006, "tagline": "Based on the incredible true story of one man's fight for freedom"},
    {"title": "The Birdcage", "released": 1996, "tagline": "Come as you are"},
    {"title": "Unforgiven", "released": 1992, "tagline": "It's a hell of a thing, killing a man"},
    {"title": "Johnny Mnemonic", "released": 1995, "tagline": "The hottest data on earth. In the coolest head in town."},
    {"title": "Cloud Atlas", "released": 2012, "tagline": "Everything is Connected"},
    {"title": "The Da Vinci Code", "released": 2006, "tagline": "Seek The Truth"},
    {"title": "V for Vendetta", "released": 2005, "tagline": "Freedom! Forever!"},
    {"title": "Speed Racer", "released": 2008, "tagline": "Speed has no limits"},
    {"title": "Apollo 13", "released": 1995, "tagline": "Houston, we have a problem."},
    {"title": "Cast Away", "released": 2000, "tagline": "At the edge of the world, his journey begins."},
    {"title": "The Green Mile", "released": 1999, "tagline": "Walk a mile you'll never forget."},
]

PERSONS: List[Dict[str, Any]] = [
    {"name": "Keanu Reeves", "born": 1964},
    {"name": "Carrie-Anne Moss", "born": 1967},
    {"name": "Laurence Fishburne", "born": 1961},
    {"name": "Hugo Weaving", "born": 1960},
    {"name": "Lilly Wachowski", "born": 1967},
    {"name": "Lana Wachowski", "born": 1965},
    {"name": "Joel Silver", "born": 1952},
    {"name": "Charlize Theron", "born": 1975},
    {"name": "Al Pacino", "born": 1940},
    {"name": "Taylor Hackford", "born": 1944},
    {"name": "Tom Cruise", "born": 1962},
    {"name": "Jack Nicholson", "born": 1937},
    {"name": "Demi Moore", "born": 1962},
    {"name": "Kevin Bacon", "born": 1958},
    {"name": "Kiefer Sutherland", "born": 1966},
    {"name": "Noah Wyle", "born": 1971},
    {"name": "Cuba Gooding Jr.", "born": 1968},
    {"name": "Kevin Pollak", "born": 1957},
    {"name": "J.T. Walsh", "born": 1943},
    {"name": "James Marshall", "born": 1967},
    {"name": "Christopher Guest", "born": 1948},
    {"name": "Rob Reiner", "born": 1947},
    {"name": "Aaron Sorkin", "born": 1961},
    {"name": "Kelly McGillis", "born": 1957},
    {"name": "Val Kilmer", "born": 1959},
    {"name": "Anthony Edwards", "born": 1962},
    {"name": "Tom Skerritt", "born": 1933},
    {"name": "Meg Ryan", "born": 1961},
    {"name": "Tony Scott", "born": 1944},
    {"name": "Renée Zellweger", "born": 1969},
    {"name": "Kelly Preston", "born": 1962},
    {"name": "Jerry O'Connell", "born": 1974},
    {"name": "Jay Mohr", "born": 1970},
    {"name": "Bonnie Hunt", "born": 1961},
    {"name": "Regina King", "born": 1971},
    {"name": "Cameron Crowe", "born": 1957},
    {"name": "River Phoenix", "born": 1970},
    {"name": "Corey Feldman", "born": 1971},
    {"name": "Wil Wheaton", "born": 1972},
    {"name": "John Cusack", "born": 1966},
    {"name": "Helen Hunt", "born": 1963},
    {"name": "Greg Kinnear", "born": 1963},
    {"name": "Robin Williams", "born": 1951},
    {"name": "Werner Herzog", "born": 1942},
    {"name": "Ethan Hawke", "born": 1970},
    {"name": "Rick Yune", "born": 1971},
    {"name": "Max von Sydow", "born": 1929},
    {"name": "Tom Hanks", "born": 1956},
    {"name": "Parker Posey", "born": 1968},
    {"name": "Dave Chappelle", "born": 1973},
    {"name": "Steve Zahn", "born": 1967},
    {"name": "Nora Ephron", "born": 1941},
    {"name": "Billy Crystal", "born": 1948},
    {"name": "Carrie Fisher", "born": 1956},
    {"name": "Gene Hackman", "born": 1930},
    {"name": "Christian Bale", "born": 1974},
    {"name": "Clint Eastwood", "born": 1930},
    {"name": "Takeshi Kitano", "born": 1947},
    {"name": "Halle Berry", "born": 1966},
    {"name": "Jim Broadbent", "born": 1949},
    {"name": "Ian McKellen", "born": 1939},
    {"name": "Audrey Tautou", "born": 1976},
    {"name": "Paul Bettany", "born": 1971},
    {"name": "Ron Howard", "born": 1954},
    {"name": "Natalie Portman", "born": 1981},
    {"name": "James McTeigue", "born": 1967},
    {"name": "Emile Hirsch", "born": 1985},
    {"name": "John Goodman", "born": 1952},
    {"name": "Susan Sarandon", "born": 1946},
    {"name": "Christina Ricci", "born": 1980},
    {"name": "Ed Harris", "born": 1950},
    {"name": "Bill Paxton", "born": 1955},
    {"name": "Gary Sinise", "born": 1955},
    {"name": "Robert Zemeckis", "born": 1951},
    {"name": "Frank Darabont", "born": 1959},
    {"name": "Michael Clarke Duncan", "born": 1957},
    {"name": "David Morse", "born": 1953},
    {"name": "Jessica Thompson", "born": 1975},
    {"name": "James Thompson", "born": 1970},
]

RELATIONSHIPS: List[Dict[str, Any]] = [
    # The Matrix
    {"source": "Keanu Reeves", "target": "The Matrix", "type": "ACTED_IN", "props": {"roles": ["Neo"]}},
    {"source": "Carrie-Anne Moss", "target": "The Matrix", "type": "ACTED_IN", "props": {"roles": ["Trinity"]}},
    {"source": "Laurence Fishburne", "target": "The Matrix", "type": "ACTED_IN", "props": {"roles": ["Morpheus"]}},
    {"source": "Hugo Weaving", "target": "The Matrix", "type": "ACTED_IN", "props": {"roles": ["Agent Smith"]}},
    {"source": "Lilly Wachowski", "target": "The Matrix", "type": "DIRECTED", "props": {}},
    {"source": "Lana Wachowski", "target": "The Matrix", "type": "DIRECTED", "props": {}},
    {"source": "Joel Silver", "target": "The Matrix", "type": "PRODUCED", "props": {}},

    # The Matrix Reloaded
    {"source": "Keanu Reeves", "target": "The Matrix Reloaded", "type": "ACTED_IN", "props": {"roles": ["Neo"]}},
    {"source": "Carrie-Anne Moss", "target": "The Matrix Reloaded", "type": "ACTED_IN", "props": {"roles": ["Trinity"]}},
    {"source": "Laurence Fishburne", "target": "The Matrix Reloaded", "type": "ACTED_IN", "props": {"roles": ["Morpheus"]}},
    {"source": "Hugo Weaving", "target": "The Matrix Reloaded", "type": "ACTED_IN", "props": {"roles": ["Agent Smith"]}},
    {"source": "Lilly Wachowski", "target": "The Matrix Reloaded", "type": "DIRECTED", "props": {}},
    {"source": "Lana Wachowski", "target": "The Matrix Reloaded", "type": "DIRECTED", "props": {}},

    # The Matrix Revolutions
    {"source": "Keanu Reeves", "target": "The Matrix Revolutions", "type": "ACTED_IN", "props": {"roles": ["Neo"]}},
    {"source": "Carrie-Anne Moss", "target": "The Matrix Revolutions", "type": "ACTED_IN", "props": {"roles": ["Trinity"]}},
    {"source": "Laurence Fishburne", "target": "The Matrix Revolutions", "type": "ACTED_IN", "props": {"roles": ["Morpheus"]}},
    {"source": "Hugo Weaving", "target": "The Matrix Revolutions", "type": "ACTED_IN", "props": {"roles": ["Agent Smith"]}},
    {"source": "Lilly Wachowski", "target": "The Matrix Revolutions", "type": "DIRECTED", "props": {}},
    {"source": "Lana Wachowski", "target": "The Matrix Revolutions", "type": "DIRECTED", "props": {}},

    # The Devil's Advocate
    {"source": "Keanu Reeves", "target": "The Devil's Advocate", "type": "ACTED_IN", "props": {"roles": ["Kevin Lomax"]}},
    {"source": "Charlize Theron", "target": "The Devil's Advocate", "type": "ACTED_IN", "props": {"roles": ["Mary Ann Lomax"]}},
    {"source": "Al Pacino", "target": "The Devil's Advocate", "type": "ACTED_IN", "props": {"roles": ["John Milton"]}},
    {"source": "Taylor Hackford", "target": "The Devil's Advocate", "type": "DIRECTED", "props": {}},

    # A Few Good Men
    {"source": "Tom Cruise", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Lt. Daniel Kaffee"]}},
    {"source": "Jack Nicholson", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Col. Nathan R. Jessep"]}},
    {"source": "Demi Moore", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Lt. Cdr. JoAnne Galloway"]}},
    {"source": "Kevin Bacon", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Capt. Jack Challee"]}},
    {"source": "Kiefer Sutherland", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Lt. Jonathan Kendrick"]}},
    {"source": "Cuba Gooding Jr.", "target": "A Few Good Men", "type": "ACTED_IN", "props": {"roles": ["Cpl. Carl Hammaker"]}},
    {"source": "Rob Reiner", "target": "A Few Good Men", "type": "DIRECTED", "props": {}},
    {"source": "Aaron Sorkin", "target": "A Few Good Men", "type": "WROTE", "props": {}},

    # Top Gun
    {"source": "Tom Cruise", "target": "Top Gun", "type": "ACTED_IN", "props": {"roles": ["Maverick"]}},
    {"source": "Kelly McGillis", "target": "Top Gun", "type": "ACTED_IN", "props": {"roles": ["Charlie"]}},
    {"source": "Val Kilmer", "target": "Top Gun", "type": "ACTED_IN", "props": {"roles": ["Iceman"]}},
    {"source": "Anthony Edwards", "target": "Top Gun", "type": "ACTED_IN", "props": {"roles": ["Goose"]}},
    {"source": "Meg Ryan", "target": "Top Gun", "type": "ACTED_IN", "props": {"roles": ["Carole"]}},
    {"source": "Tony Scott", "target": "Top Gun", "type": "DIRECTED", "props": {}},

    # Jerry Maguire
    {"source": "Tom Cruise", "target": "Jerry Maguire", "type": "ACTED_IN", "props": {"roles": ["Jerry Maguire"]}},
    {"source": "Cuba Gooding Jr.", "target": "Jerry Maguire", "type": "ACTED_IN", "props": {"roles": ["Rod Tidwell"]}},
    {"source": "Renée Zellweger", "target": "Jerry Maguire", "type": "ACTED_IN", "props": {"roles": ["Dorothy Boyd"]}},
    {"source": "Bonnie Hunt", "target": "Jerry Maguire", "type": "ACTED_IN", "props": {"roles": ["Laurel Boyd"]}},
    {"source": "Cameron Crowe", "target": "Jerry Maguire", "type": "DIRECTED", "props": {}},
    {"source": "Cameron Crowe", "target": "Jerry Maguire", "type": "PRODUCED", "props": {}},

    # Stand By Me
    {"source": "Wil Wheaton", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Gordie Lachance"]}},
    {"source": "River Phoenix", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Chris Chambers"]}},
    {"source": "Jerry O'Connell", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Vern Tessio"]}},
    {"source": "Corey Feldman", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Teddy Duchamp"]}},
    {"source": "John Cusack", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Denny Lachance"]}},
    {"source": "Kiefer Sutherland", "target": "Stand By Me", "type": "ACTED_IN", "props": {"roles": ["Ace Merrill"]}},
    {"source": "Rob Reiner", "target": "Stand By Me", "type": "DIRECTED", "props": {}},

    # As Good as It Gets
    {"source": "Jack Nicholson", "target": "As Good as It Gets", "type": "ACTED_IN", "props": {"roles": ["Melvin Udall"]}},
    {"source": "Helen Hunt", "target": "As Good as It Gets", "type": "ACTED_IN", "props": {"roles": ["Carol Connelly"]}},
    {"source": "Greg Kinnear", "target": "As Good as It Gets", "type": "ACTED_IN", "props": {"roles": ["Simon Bishop"]}},
    {"source": "Cuba Gooding Jr.", "target": "As Good as It Gets", "type": "ACTED_IN", "props": {"roles": ["Frank Sachs"]}},

    # You've Got Mail
    {"source": "Tom Hanks", "target": "You've Got Mail", "type": "ACTED_IN", "props": {"roles": ["Joe Fox"]}},
    {"source": "Meg Ryan", "target": "You've Got Mail", "type": "ACTED_IN", "props": {"roles": ["Kathleen Kelly"]}},
    {"source": "Greg Kinnear", "target": "You've Got Mail", "type": "ACTED_IN", "props": {"roles": ["Frank Navasky"]}},
    {"source": "Dave Chappelle", "target": "You've Got Mail", "type": "ACTED_IN", "props": {"roles": ["Kevin Jackson"]}},
    {"source": "Steve Zahn", "target": "You've Got Mail", "type": "ACTED_IN", "props": {"roles": ["George Pappas"]}},
    {"source": "Nora Ephron", "target": "You've Got Mail", "type": "DIRECTED", "props": {}},

    # Sleepless in Seattle
    {"source": "Tom Hanks", "target": "Sleepless in Seattle", "type": "ACTED_IN", "props": {"roles": ["Sam Baldwin"]}},
    {"source": "Meg Ryan", "target": "Sleepless in Seattle", "type": "ACTED_IN", "props": {"roles": ["Annie Reed"]}},
    {"source": "Nora Ephron", "target": "Sleepless in Seattle", "type": "DIRECTED", "props": {}},

    # When Harry Met Sally
    {"source": "Billy Crystal", "target": "When Harry Met Sally", "type": "ACTED_IN", "props": {"roles": ["Harry Burns"]}},
    {"source": "Meg Ryan", "target": "When Harry Met Sally", "type": "ACTED_IN", "props": {"roles": ["Sally Albright"]}},
    {"source": "Carrie Fisher", "target": "When Harry Met Sally", "type": "ACTED_IN", "props": {"roles": ["Marie"]}},
    {"source": "Rob Reiner", "target": "When Harry Met Sally", "type": "DIRECTED", "props": {}},
    {"source": "Nora Ephron", "target": "When Harry Met Sally", "type": "WROTE", "props": {}},

    # That Thing You Do
    {"source": "Tom Hanks", "target": "That Thing You Do", "type": "ACTED_IN", "props": {"roles": ["Mr. White"]}},
    {"source": "Charlize Theron", "target": "That Thing You Do", "type": "ACTED_IN", "props": {"roles": ["Tina"]}},
    {"source": "Tom Hanks", "target": "That Thing You Do", "type": "DIRECTED", "props": {}},

    # The Replacements
    {"source": "Keanu Reeves", "target": "The Replacements", "type": "ACTED_IN", "props": {"roles": ["Shane Falco"]}},
    {"source": "Gene Hackman", "target": "The Replacements", "type": "ACTED_IN", "props": {"roles": ["Jimmy McGinty"]}},

    # Rescue Dawn
    {"source": "Christian Bale", "target": "Rescue Dawn", "type": "ACTED_IN", "props": {"roles": ["Dieter Dengler"]}},
    {"source": "Steve Zahn", "target": "Rescue Dawn", "type": "ACTED_IN", "props": {"roles": ["Duane"]}},
    {"source": "Werner Herzog", "target": "Rescue Dawn", "type": "DIRECTED", "props": {}},

    # Unforgiven
    {"source": "Clint Eastwood", "target": "Unforgiven", "type": "ACTED_IN", "props": {"roles": ["Bill Munny"]}},
    {"source": "Gene Hackman", "target": "Unforgiven", "type": "ACTED_IN", "props": {"roles": ["Little Bill Daggett"]}},
    {"source": "Clint Eastwood", "target": "Unforgiven", "type": "DIRECTED", "props": {}},

    # Johnny Mnemonic
    {"source": "Keanu Reeves", "target": "Johnny Mnemonic", "type": "ACTED_IN", "props": {"roles": ["Johnny Mnemonic"]}},
    {"source": "Takeshi Kitano", "target": "Johnny Mnemonic", "type": "ACTED_IN", "props": {"roles": ["Takahashi"]}},

    # Cloud Atlas
    {"source": "Tom Hanks", "target": "Cloud Atlas", "type": "ACTED_IN", "props": {"roles": ["Zachry", "Dr. Henry Goose"]}},
    {"source": "Hugo Weaving", "target": "Cloud Atlas", "type": "ACTED_IN", "props": {"roles": ["Bill Smoke", "Haskell Moore"]}},
    {"source": "Halle Berry", "target": "Cloud Atlas", "type": "ACTED_IN", "props": {"roles": ["Luisa Rey"]}},
    {"source": "Jim Broadbent", "target": "Cloud Atlas", "type": "ACTED_IN", "props": {"roles": ["Vyvyan Ayrs"]}},
    {"source": "Lilly Wachowski", "target": "Cloud Atlas", "type": "DIRECTED", "props": {}},
    {"source": "Lana Wachowski", "target": "Cloud Atlas", "type": "DIRECTED", "props": {}},

    # The Da Vinci Code
    {"source": "Tom Hanks", "target": "The Da Vinci Code", "type": "ACTED_IN", "props": {"roles": ["Dr. Robert Langdon"]}},
    {"source": "Ian McKellen", "target": "The Da Vinci Code", "type": "ACTED_IN", "props": {"roles": ["Sir Leigh Teabing"]}},
    {"source": "Audrey Tautou", "target": "The Da Vinci Code", "type": "ACTED_IN", "props": {"roles": ["Sophie Neveu"]}},
    {"source": "Paul Bettany", "target": "The Da Vinci Code", "type": "ACTED_IN", "props": {"roles": ["Silas"]}},
    {"source": "Ron Howard", "target": "The Da Vinci Code", "type": "DIRECTED", "props": {}},

    # V for Vendetta
    {"source": "Hugo Weaving", "target": "V for Vendetta", "type": "ACTED_IN", "props": {"roles": ["V"]}},
    {"source": "Natalie Portman", "target": "V for Vendetta", "type": "ACTED_IN", "props": {"roles": ["Evey Hammond"]}},
    {"source": "James McTeigue", "target": "V for Vendetta", "type": "DIRECTED", "props": {}},
    {"source": "Lilly Wachowski", "target": "V for Vendetta", "type": "PRODUCED", "props": {}},
    {"source": "Lana Wachowski", "target": "V for Vendetta", "type": "PRODUCED", "props": {}},
    {"source": "Joel Silver", "target": "V for Vendetta", "type": "PRODUCED", "props": {}},

    # Speed Racer
    {"source": "Emile Hirsch", "target": "Speed Racer", "type": "ACTED_IN", "props": {"roles": ["Speed Racer"]}},
    {"source": "John Goodman", "target": "Speed Racer", "type": "ACTED_IN", "props": {"roles": ["Pops Racer"]}},
    {"source": "Susan Sarandon", "target": "Speed Racer", "type": "ACTED_IN", "props": {"roles": ["Mom Racer"]}},
    {"source": "Christina Ricci", "target": "Speed Racer", "type": "ACTED_IN", "props": {"roles": ["Trixie"]}},
    {"source": "Lilly Wachowski", "target": "Speed Racer", "type": "DIRECTED", "props": {}},
    {"source": "Lana Wachowski", "target": "Speed Racer", "type": "DIRECTED", "props": {}},
    {"source": "Joel Silver", "target": "Speed Racer", "type": "PRODUCED", "props": {}},

    # Apollo 13
    {"source": "Tom Hanks", "target": "Apollo 13", "type": "ACTED_IN", "props": {"roles": ["Jim Lovell"]}},
    {"source": "Kevin Bacon", "target": "Apollo 13", "type": "ACTED_IN", "props": {"roles": ["Jack Swigert"]}},
    {"source": "Ed Harris", "target": "Apollo 13", "type": "ACTED_IN", "props": {"roles": ["Gene Kranz"]}},
    {"source": "Bill Paxton", "target": "Apollo 13", "type": "ACTED_IN", "props": {"roles": ["Fred Haise"]}},
    {"source": "Gary Sinise", "target": "Apollo 13", "type": "ACTED_IN", "props": {"roles": ["Ken Mattingly"]}},
    {"source": "Ron Howard", "target": "Apollo 13", "type": "DIRECTED", "props": {}},

    # Cast Away
    {"source": "Tom Hanks", "target": "Cast Away", "type": "ACTED_IN", "props": {"roles": ["Chuck Noland"]}},
    {"source": "Helen Hunt", "target": "Cast Away", "type": "ACTED_IN", "props": {"roles": ["Kelly Frears"]}},
    {"source": "Robert Zemeckis", "target": "Cast Away", "type": "DIRECTED", "props": {}},

    # The Green Mile
    {"source": "Tom Hanks", "target": "The Green Mile", "type": "ACTED_IN", "props": {"roles": ["Paul Edgecomb"]}},
    {"source": "Michael Clarke Duncan", "target": "The Green Mile", "type": "ACTED_IN", "props": {"roles": ["John Coffey"]}},
    {"source": "David Morse", "target": "The Green Mile", "type": "ACTED_IN", "props": {"roles": ["Brutus 'Brutal' Howell"]}},
    {"source": "Bonnie Hunt", "target": "The Green Mile", "type": "ACTED_IN", "props": {"roles": ["Jan Edgecomb"]}},
    {"source": "Frank Darabont", "target": "The Green Mile", "type": "DIRECTED", "props": {}},

    # Reviews and Social
    {"source": "Jessica Thompson", "target": "The Matrix", "type": "REVIEWED", "props": {"summary": "An amazing sci-fi classic", "rating": 95}},
    {"source": "James Thompson", "target": "The Matrix", "type": "REVIEWED", "props": {"summary": "Mind bending visuals and concept", "rating": 90}},
    {"source": "Jessica Thompson", "target": "A Few Good Men", "type": "REVIEWED", "props": {"summary": "Great courtroom drama", "rating": 88}},
    {"source": "Jessica Thompson", "target": "James Thompson", "type": "FOLLOWS", "props": {}},
]
